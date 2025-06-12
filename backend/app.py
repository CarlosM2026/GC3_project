"""
CS-410: Generates spectrograms from uploaded files and stores them in MongoDB
@file app.py
@authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
@collaborators None
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pymongo import MongoClient
from bson import ObjectId
from gridfs import GridFS
from gridfs import errors as gridfs_errors
from SigMF import SigMF
from FileData import FileData
import csv
import os
import json
from airview import Plugin
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import subprocess, threading

import matplotlib
# Use the Agg backend for Matplotlib to avoid using any X server
matplotlib.use('Agg')

"""
 * Creates and configures the Flask application.
 * @return The configured Flask application.
"""
def create_app():
    # Initialize the Flask application
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # MongoDB setup using GridFS
    client = MongoClient(
        "mongodb://generalUser:password@150.209.91.78:27017/frequencyDB"
    )

    db = client['files_db']
    fs = GridFS(db)

    # NOTE: For those with deuteranopia, change cmap='viridis' to cmap='accessible_cmap' in
    #       plot_spectrogram() and generate_data() to use a color palette that is more accessible 
    #       for colorblind users.
    SPEC_HEX_COLORS = ["#e4ff7a", "#ffe81a", "#ffbd00", "#ffa000", "#fc7f00"] # CITE: https://github.com/wistia/heatmap-palette
    rgb_colors = [to_rgb(color) for color in SPEC_HEX_COLORS] # Convert hex to RGB
    custom_cmap = LinearSegmentedColormap.from_list('accessible_cmap', rgb_colors, N=256) # Create a custom colormap
    plt.colormaps.register(cmap=custom_cmap) #Switched to newer version for colormap (original line below)
    #plt.register_cmap(cmap=custom_cmap) # Register the colormap with Matplotlib

    #gui 
    @app.route('/launch-sdr-gui', methods=['POST'])
    def launch_sdr_gui():
        data = request.json
        serial = data.get('serial')
        type_ = data.get('type_')
        connected = data.get('connected')

        def _launch():
            subprocess.run(['python3', 'bridged-gui.py', serial, type_, str(int(connected))])

        threading.Thread(target=_launch, daemon=True).start()
        print("launched")
        return jsonify(status='launched')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Uploads files, generates plots, stores in MongoDB."""
        if 'cfile' not in request.files or 'metaFile' not in request.files:
            return jsonify({'error': 'Both .cfile and .sigmf-meta files are required'}), 400
        
        run_airview     = request.form.get('runAirview',  'true').lower()  in ('1','true','yes','y')
        run_download    = request.form.get('downloadCSV','false').lower() in ('1','true','yes','y')
        auto_params     = request.form.get('autoParams','false').lower()  in ('1','true','yes','y')
        # pull manual overrides (fall back to Plugin defaults)
        beta_manual     = float(request.form.get('beta',     Plugin.beta))
        scale_manual    = int(  request.form.get('scale',    Plugin.scale))
        print(f"[UPLOAD] runAirview={run_airview}, downloadCSV={run_download}, autoParams={auto_params}, beta={beta_manual}, scale={scale_manual}")

        cfile, metafile = request.files['cfile'], request.files['metaFile']
        original_name = cfile.filename.replace('.cfile', '')


        try:
            # Pass the actual file object to SigMF, NOT a string
            sigmf_metadata = SigMF(metafile)
        except Exception as e:
            return jsonify({'error': f'Failed to parse metadata: {str(e)}'}), 400

        # Read cfile contents once
        cfile_bytes = cfile.read()
        iq_data = np.frombuffer(cfile_bytes, dtype=np.complex64)

        # instantiate Plugin with either auto‑opt or manual params
        plugin = Plugin(
            sample_rate=sigmf_metadata.sample_rate,
            center_freq=sigmf_metadata.center_frequency,
            run_parameter_optimization = 'y' if auto_params else 'n',
            beta  = beta_manual,
            scale = scale_manual
        )
        if run_airview:
            result = plugin.run(iq_data)
            if auto_params:
                # AirVIEW returns the best [beta, scale]
                trained_beta, trained_scale = result.get("airview_beta_scale", [beta_manual, scale_manual])
                airview_annotations = []
            else:
                airview_annotations = result.get("airview_annotations", [])
                trained_beta, trained_scale = beta_manual, scale_manual
        else:
            airview_annotations, trained_beta, trained_scale = [], beta_manual, scale_manual


        plot_ids, Pxx, freqs, bins = generate_plots(original_name, iq_data, sigmf_metadata)
        if run_download:
            pxx_csv_file_id = save_pxx_csv(original_name, Pxx, freqs, bins)
        else:
            pxx_csv_file_id = None
            
        # Save the metadata file in GridFS
        metafile.seek(0)  # Reset file pointer before saving
        meta_file_id = fs.put(metafile.read(), filename=f"{original_name}.sigmf-meta")

        # Store metadata file ID in file_records
        file_data = FileData(original_name, sigmf_metadata, pxx_csv_file_id, plot_ids, freqs, bins, 1024, airview_annotations)
        file_data.meta_file_id = meta_file_id  # Save metadata file ID
        file_data.airview_annotations = airview_annotations  # Save airview annotations
        file_record_id = db.file_records.insert_one(file_data.__dict__).inserted_id

        encoded_spectrogram = base64.b64encode(fs.get(plot_ids["spectrogram"]).read()).decode('utf-8')
        
        print("sending json to frontend")
        print("AIRVIEW ANNOTATIONS: ")
        print(airview_annotations)

        return jsonify({
            'spectrogram': encoded_spectrogram,
            'file_id':    str(file_record_id),
            'message':    'All files uploaded and saved successfully',
            'airview_annotations':       airview_annotations,
            'beta_used':         trained_beta,
            'scale_used':        trained_scale, 
            'max_time': file_data.max_time,
            'min_freq': file_data.min_freq,
            'max_freq': file_data.max_freq,
            'airview_annotations': file_data.airview_annotations,
        })
    
            
    @app.route('/save-file', methods=['POST'])
    def save_file():
        try:
            data = request.json
            file_id = data.get('file_id')
            annotations = data.get('annotations', [])
            if not file_id:
                return jsonify({"error": "File ID is required"}), 400
            # Update the file record with annotations
            db.file_records.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"annotations": annotations}}
            )
            return jsonify({"message": "File saved successfully"})
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/<file_id>/rename_file', methods=['PUT'])
    def rename_file(file_id):
        """Renames file that is displayed under 'Saved Files'"""
        try:
            data = request.json
            new_filename = data.get('filename')
            
            if not new_filename:
                return jsonify({"error": "New filename is required"}), 400

            # Update filename in file_records
            result = db.file_records.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"filename": new_filename}}
            )

            if result.matched_count == 0:
                return jsonify({"error": "File not found"}), 404

            return jsonify({"message": "File renamed successfully"})

        except Exception as e:
            print(f"Error renaming file: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route('/file/<file_id>/csv', methods=['GET'])
    def download_pxx_csv(file_id):
        """Serves the Pxx CSV as a downloadable file."""
        from bson import ObjectId
        # find our record
        rec = db.file_records.find_one({"_id": ObjectId(file_id)})
        if not rec or "csv_file_id" not in rec:
            return jsonify({"error":"CSV not found"}), 404

        try:
            grid_out = fs.get(ObjectId(rec["csv_file_id"]))
            csv_bytes = grid_out.read()
        except gridfs_errors.NoFile:
            return jsonify({"error":"CSV missing in GridFS"}), 404

        fname = f"{rec['filename']}_pxx.csv"
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={fname}"
            }
        )

    def generate_plots(original_name, iq_data, sigmf_metadata):
        """Generates and stores plots in GridFS."""
        plots = {}

        # Debug: Starting plot generation
        print(f"Generating plots for {original_name}...")

        # Generate spectrogram and get Pxx, freqs, bins
        fig, Pxx, freqs, bins = plot_spectrogram(iq_data, sigmf_metadata)
        spectrogram_file_id = save_plot(fig, f"{original_name}_spectrogram.png")
        plots["spectrogram"] = spectrogram_file_id
        print(f"Saved Spectrogram: {spectrogram_file_id}")

        # Generate time domain plot
        fig = plot_time_domain(iq_data, sigmf_metadata)
        plots["time_domain"] = save_plot(fig, f"{original_name}_time_domain.png")
        print(f"Saved Time Domain Plot: {plots['time_domain']}")

        # Generate frequency domain (FFT) plot
        fig = plot_freq_domain(iq_data, sigmf_metadata)
        plots["freq_domain"] = save_plot(fig, f"{original_name}_freq_domain.png")
        print(f"Saved Frequency Domain Plot: {plots['freq_domain']}")

        # Generate IQ plot (Constellation Diagram)
        fig = plot_iq(iq_data)
        plots["iq_plot"] = save_plot(fig, f"{original_name}_iq_plot.png")
        print(f"Saved IQ Plot: {plots['iq_plot']}")

        # Debug: Finished plot generation
        print(f"All plots generated and saved for {original_name}.")

        return plots, Pxx, freqs, bins

    def save_plot(fig, filename):
        """Saves a given Matplotlib figure to GridFS."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return fs.put(buf.getvalue(), filename=filename)

    def plot_time_domain(iq_data, sigmf_metadata):
        """Generates the time-domain plot."""
        fig, ax = plt.subplots(figsize=(8, 4))
        time_axis = np.arange(len(iq_data)) / sigmf_metadata.sample_rate
        ax.plot(time_axis[:1000], iq_data[:1000].real, label="Real")
        ax.plot(time_axis[:1000], iq_data[:1000].imag, label="Imaginary", linestyle='dashed')
        ax.set_title("Time Domain Signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend()
        return fig
    #can be used for waterfall
    def plot_waterfall(iq_data, sigmf_metadata):
        
        # fig, ax = plt.subplots(figsize=(8, 4))
        # freq_axis = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1 / sigmf_metadata.sample_rate))
        # time_axis = np.arange(len(iq_data)) / sigmf_metadata.sample_rate
        # fft_spectrum = np.fft.fftshift(np.fft.fft(iq_data))
        # temp_mag = 20 * np.log10(np.abs(fft_spectrum))
        # ax.plot(time_axis[:1000], freq_axis)

        return fig
    def plot_freq_domain(iq_data, sigmf_metadata):
        """Generates the frequency-domain (FFT) plot."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fft_spectrum = np.fft.fftshift(np.fft.fft(iq_data))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1 / sigmf_metadata.sample_rate))
        ax.plot(freq_axis, 20 * np.log10(np.abs(fft_spectrum)), color='red')
        ax.set_title("Frequency Domain (FFT)")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")
        return fig

    def plot_iq(iq_data):
        """Generates the IQ plot (constellation diagram)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(iq_data[:5000].real, iq_data[:5000].imag, alpha=0.5, s=2)
        ax.set_title("IQ Plot (Constellation Diagram)")
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        return fig

    def plot_spectrogram(iq_data, sigmf_metadata):
        """Generates the spectrogram and returns Pxx, freqs, bins."""
        fig, ax = plt.subplots(figsize=(8, 4.8))
        # Generate the spectrogram
        Pxx, freqs, bins, im = ax.specgram(
            iq_data,
            Fs=sigmf_metadata.sample_rate,
            Fc=sigmf_metadata.center_frequency,
        )
        # Overlay image representation of Pxx (Power Spectral Density)
        ax.imshow(10 * np.log10(Pxx.T), aspect='auto', extent=[freqs[0], freqs[-1], bins[-1], 0], cmap='viridis')
        # Set plot labels
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Time [s]")
        ax.set_title("Spectrogram")
        return fig, Pxx, freqs, bins

    def save_pxx_csv(original_name, Pxx, freqs, bins):
        """Saves the Pxx matrix as a CSV in GridFS."""
        pxx_csv_data = io.StringIO()
        csv_writer = csv.writer(pxx_csv_data)
        
        csv_writer.writerow(["Frequency (Hz)"] + bins.tolist())
        for i, freq in enumerate(freqs):
            csv_writer.writerow([freq] + Pxx[i].tolist())

        pxx_csv_data.seek(0)
        print("Generated csv")

        return fs.put(pxx_csv_data.getvalue().encode(), filename=f"{original_name}_pxx.csv")
    
    @app.route('/file/<file_id>', methods=['DELETE'])
    def delete_file(file_id):
        """Deletes a file and its associated data (plots, Pxx file, metadata)."""
        try:
            print("DELETING.....")
            
            # Retrieve the file record from the database
            file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
            if not file_record:
                print("file not found")
                return jsonify({"error": "File not found"}), 404
            
            print("File Record Attributes:")
            for key, value in file_record.items():
                print(f"{key}: {value}")

            # Delete PXX file
            try:
                print("Attempting to delete associated files...")
                if file_record['csv_file_id'] != "None":
                    print(f"Deleting CSV file: {file_record['csv_file_id']}")
                    fs.delete(ObjectId(file_record["csv_file_id"]))
                    print("CSV file deleted")
                
                if file_record['raw_data_file_id'] is not None:
                    print(f"Deleting raw data file: {file_record['raw_data_file_id']}")
                    fs.delete(ObjectId(file_record["iq_plot_file_id"]))
                    print("IQ plot file deleted")

                if "iq_plot_file_id" in file_record:
                    print(f"Deleting IQ plot file: {file_record['iq_plot_file_id']}")
                    fs.delete(ObjectId(file_record["iq_plot_file_id"]))
                    print("IQ plot file deleted")
                
                if "spectrogram_file_id" in file_record:
                    print(f"Deleting Spectrogram file: {file_record['spectrogram_file_id']}")
                    fs.delete(ObjectId(file_record["spectrogram_file_id"]))
                    print("Spectrogram file deleted")
                
                if "time_domain_file_id" in file_record:
                    print(f"Deleting Time Domain file: {file_record['time_domain_file_id']}")
                    fs.delete(ObjectId(file_record["time_domain_file_id"]))
                    print("Time Domain file deleted")
                
                if "freq_domain_file_id" in file_record:
                    print(f"Deleting Frequency Domain file: {file_record['freq_domain_file_id']}")
                    fs.delete(ObjectId(file_record["freq_domain_file_id"]))
                    print("Frequency Domain file deleted")
                
                if "meta_file_id" in file_record:
                    print(f"Deleting Metadata file: {file_record['meta_file_id']}")
                    fs.delete(ObjectId(file_record["meta_file_id"]))
                    print("Metadata file deleted")
                
            except gridfs_errors.NoFile:
                print(f"file not found.")


            db.file_records.delete_one({"_id": ObjectId(file_id)})

            return jsonify({"message": f"File with ID {file_id} and its associated data deleted successfully."})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/files', methods=['GET'])
    def get_files():
        """Lists all stored filenames."""
        try:
            print("GETTING FILES")
            files = list(db.file_records.find({}, {"filename": 1}))
            print(files)
            file_list = [{"_id": str(file["_id"]), "filename": file["filename"]} for file in files]
            return jsonify({"files": file_list})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/file/<file_id>/spectrogram', methods=['GET'])
    def get_file_spectrogram(file_id):
        """Retrieves the spectrogram PNG from GridFS using the saved file ID."""
        try:
            if not ObjectId.is_valid(file_id):
                return jsonify({'error': 'Invalid file ID format'}), 400  

            file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
            if not file_record:
                return jsonify({'error': 'File not found'}), 404

            spectrogram_file = fs.get(ObjectId(file_record["spectrogram_file_id"]))
            return jsonify({'image': base64.b64encode(spectrogram_file.read()).decode('utf-8')})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        

    @app.route('/file/<file_id>/data', methods=['GET'])
    def get_file_data(file_id):
        """Fetches fileData for a given file ID."""
        try:
            if not ObjectId.is_valid(file_id):
                return jsonify({'error': 'Invalid file ID format'}), 400

            file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
            if not file_record:
                return jsonify({'error': 'File not found'}), 404
            
            if "max_time" not in file_record:
                return jsonify({'error': 'max_time not found in record'}), 400
            if "min_freq" not in file_record:
                return jsonify({'error': 'min_freq not found in record'}), 400
            if "max_freq" not in file_record:
                return jsonify({'error': 'max_freq not found in record'}), 400
            
            print(f"max time: {file_record.get('max_time')}")
            print(f"min freq: {file_record.get('min_freq')}")
            print(f"max freq: {file_record.get('max_freq')}")
            print(f"annotations: {file_record.get('annotations', [])}")
            
            # Return relevant fields from fileData
            return jsonify({
                'max_time': file_record.get('max_time'),
                'min_freq': file_record.get('min_freq'),
                'max_freq': file_record.get('max_freq'),
                'annotations': file_record.get('annotations', []), 
                'airview_annotations': file_record.get("airview_annotations", [])
            })
        except Exception as e:
            print(f"Error fetching file data: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/refresh', methods=['POST'])
    def refresh_files():
        """Clears all saved files and metadata."""
        try:
            db.file_records.delete_many({})
            for file in fs.find():
                fs.delete(file._id)
            return jsonify({'message': 'All files have been cleared.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/file/<file_id>/<plot_type>', methods=['GET'])
    def get_file_plot(file_id, plot_type):
        """Retrieves a requested plot (spectrogram, time domain, frequency domain, or IQ plot) from GridFS."""
        try:
            if not ObjectId.is_valid(file_id):
                return jsonify({'error': 'Invalid file ID format'}), 400  

            file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
            if not file_record or f"{plot_type}_file_id" not in file_record:
                return jsonify({'error': f'{plot_type} file not found'}), 404

            # Fetch the file from GridFS
            plot_file = fs.get(ObjectId(file_record[f"{plot_type}_file_id"]))
            return jsonify({'image': base64.b64encode(plot_file.read()).decode('utf-8')})

        except gridfs_errors.NoFile:
            return jsonify({'error': f'{plot_type} file does not exist in GridFS'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    import json

    @app.route('/metadata/<file_id>', methods=['GET'])
    def get_metadata(file_id):
        """Fetch metadata for a given file."""
        if not ObjectId.is_valid(file_id):
            return jsonify({'error': 'Invalid file ID format'}), 400  

        file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
        if not file_record:
            return jsonify({'error': 'File not found'}), 404

        # Ensure meta_file_id exists in file_record
        if "meta_file_id" not in file_record:
            return jsonify({'error': 'meta_file_id not found in record'}), 400

        try:
            # Fetch the metadata file from GridFS
            meta_file = fs.get(ObjectId(file_record["meta_file_id"]))

            # Read and decode metadata
            meta_content = meta_file.read().decode('utf-8')  # Keep this as a string!

            # Pass the JSON string to SigMF (not a dict!)
            sigmf_metadata = SigMF(io.StringIO(meta_content))

        except gridfs_errors.NoFile:
            return jsonify({'error': 'Metadata file not found in GridFS'}), 404
        except json.JSONDecodeError:
            return jsonify({'error': 'Failed to parse metadata JSON'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing SigMF: {str(e)}'}), 500

        # Return extracted metadata
        metadata = {
            "datatype": sigmf_metadata.datatype,
            "sample_rate": sigmf_metadata.sample_rate,
            "author": sigmf_metadata.author,
            "hardware": sigmf_metadata.hardware,
            "offset": sigmf_metadata.offset,
            "recorder": sigmf_metadata.recorder,
            "datetime": sigmf_metadata.datetime,
            "center_frequency": sigmf_metadata.center_frequency,
            "sample_start": sigmf_metadata.sample_start,
        }

        return jsonify(metadata)
    
    # NOTE: Visualization Limitation
    # The spectrogram visualization may not visibly reflect changes to noise_mean due to 
    # matplotlib's automatic color scaling. plt.imshow() automatically rescales the colormap
    # to fit the full range of data in the matrix. This means that changing noise_mean shifts
    # all values in the matrix but the colors are automatically rescaled to use the same 
    # visual range regardless of the absolute values. Hoever,the CSV data contains the 
    # correct numerical values with the specified noise_mean.
    def generate_data(rows, cols, num_transmitters, transmitter_mean, transmitter_sd, noise_mean, noise_sd, bandwidth, active_time, placement_method):
        matrix = np.random.normal(loc=noise_mean, scale=noise_sd, size=(rows, cols))
        """Generate a syntehtic data matrix with Gaussian noise and inject transmitter signals."""

        transmitters = []
        center_freq = cols // 2  # Center frequency bin

        def is_overlapping(start_time, start_freq):
            """Check if the new transmitter overlaps with existing ones."""
            for t_start, f_start in transmitters:
                if (start_time < t_start + active_time and start_time + active_time > t_start and
                    start_freq < f_start + bandwidth and start_freq + bandwidth > f_start):
                    return True
            return False

        if placement_method == "random":
            for _ in range(num_transmitters):
                while True:
                    start_time = np.random.randint(0, rows - active_time + 1)
                    start_freq = center_freq - (bandwidth // 2)  # Center the transmitter around the middle frequency
                    if not is_overlapping(start_time, start_freq):
                        transmitters.append((start_time, start_freq))
                        break

        elif placement_method == "equally_spaced":
            if num_transmitters == 1:
                # Handle single transmitter case - place it randomly
                start_time = np.random.randint(0, rows - active_time + 1)
                start_freq = center_freq - (bandwidth // 2)
                transmitters.append((start_time, start_freq))
            else:
                # Place the first transmitter randomly in the top third of the matrix
                # NOTE: Change as needed
                top_third = (rows - active_time) // 3
                min_position = max(0, top_third // 2)  # Ensure some margin from the very top
                first_transmitter_pos = np.random.randint(min_position, min_position + top_third)
                
                # Add the first transmitter
                transmitters.append((first_transmitter_pos, center_freq - (bandwidth // 2)))
                
                # Calculate available space from first transmitter to bottom of matrix
                available_space = rows - first_transmitter_pos - active_time
                
                # Calculate total number of gaps needed
                total_gaps = num_transmitters
                
                # Calculate the size of each gap to ensure equal spacing
                gap_size = available_space // total_gaps
                
                # Place remaining transmitters with equal gaps
                for i in range(1, num_transmitters):
                    start_time = first_transmitter_pos + active_time + (i * gap_size)
                    start_freq = center_freq - (bandwidth // 2)
                    transmitters.append((start_time, start_freq))

        # Inject the transmitter signal
        for start_time, start_freq in transmitters:
            for t in range(start_time, start_time + active_time):
                for f in range(start_freq, start_freq + bandwidth):
                    if 0 <= t < rows and 0 <= f < cols:  # Ensure we're within matrix bounds
                        signal = np.random.normal(loc=transmitter_mean, scale=transmitter_sd)
                        matrix[t][f] += signal

        # Generate plot as base64
        plt.figure(figsize=(10, 6))
        plt.imshow(matrix, aspect='auto', cmap='viridis')
        plt.title('Generated Data Matrix')
        plt.xlabel("Frequency")
        plt.ylabel("Time")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Generate CSV data
        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)
        
        # Add header row with column indices
        header = ['Row/Col'] + [str(i) for i in range(cols)]
        csv_writer.writerow(header)
        
        # Add data rows
        for i in range(rows):
            row_data = [str(i)] + [str(float(x)) for x in matrix[i]]
            csv_writer.writerow(row_data)
        
        csv_data.seek(0)
        csv_string = csv_data.getvalue()
        
        return plot_data, csv_string
    
    @app.route('/file/<file_id>/calculated_statistics', methods=['GET'])
    def get_calculated_statistics(file_id):
        """Calculate and return FFT size, sampling frequency, frequency resolution, and row duration."""
        if not ObjectId.is_valid(file_id):
            return jsonify({'error': 'Invalid file ID format'}), 400

        file_record = db.file_records.find_one({"_id": ObjectId(file_id)})
        if not file_record:
            return jsonify({'error': 'File not found'}), 404

        try:
            # Load the metadata
            meta_file = fs.get(ObjectId(file_record["meta_file_id"]))
            meta_content = meta_file.read().decode('utf-8')
            sigmf_metadata = SigMF(io.StringIO(meta_content))

            # Parameters
            sample_rate = sigmf_metadata.sample_rate
            N = 256
            freq_resolution = sample_rate / N
            row_duration = N / sample_rate

            
            return jsonify({
                "fft_size": N,
                "sampling_frequency": sample_rate,
                "frequency_resolution": freq_resolution,
                "row_duration": row_duration
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/generate', methods=['POST'])
    def generate_data_endpoint():
        """High level function for generating spectrogram"""
        data = request.json
        rows = data['rows']
        cols = data['cols']
        num_transmitters = data['numTransmitters']
        transmitter_mean = data['transmitterMean']
        transmitter_sd = data['transmitterSd']
        noise_mean = data['noiseMean']
        noise_sd = data['noiseSd']
        bandwidth = data['bandwidth']
        active_time = data['activeTime']
        placement_method = data['placementMethod']

        try:
            plot_data, csv_data = generate_data(rows, cols, num_transmitters, transmitter_mean, transmitter_sd, noise_mean, noise_sd,
                                    bandwidth, active_time, placement_method)
            return jsonify({
                'plot': plot_data,
                'csv': csv_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)