# GC3

GC3 is a comprehensive web application for uploading, visualizing, and analyzing radio frequency (RF) signal data. It provides an intuitive interface for working with SigMF-compatible files and includes powerful analysis tools.

## Features

- **File Management**: Upload and organize SigMF-compatible `.cfile` and `.sigmf-meta` files
- **Visualizations**: Generate spectrograms, time domain plots, frequency domain plots, and IQ plots
- **Signal Analysis**: Detect and analyze transmissions using AirVIEW integration
- **Annotation System**: Create, edit, and manage annotations on spectrograms
- **Metadata Extraction**: View and analyze file metadata and calculated statistics
- **Synthetic Data**: Generate test data with configurable parameters
- **Multi-tab Interface**: Work with multiple files simultaneously

## Prerequisites

- **Node.js and npm** (for frontend)
- **Python 3.8+** (for backend)
- **Web browser** (Chrome, Firefox, Safari, or Edge recommended)

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install required Python packages:
   ```bash
   pip install Flask flask-cors pymongo matplotlib numpy scipy pillow python-dotenv
   ```
   
### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

Run backend and frontend in separate terminals:

1. **Start the backend server**:
   ```bash
   cd backend
   python app.py
   ```

2. **Start the frontend development server**:
   ```bash
   cd frontend
   npm run dev
   ```

3. Access the application at [http://localhost:5173](http://localhost:5173)

## Usage Guide

### Uploading Files

1. Click on "Choose .cfile" to select a data file
2. Click on "Choose .sigmf" to select the corresponding metadata file
3. Optional: Toggle "Run AirVIEW" for transmission detection
4. Optional: Toggle "Download CSV" to save processed data
5. Click "Upload" to process the files

### Using Synthetic Data Generator

1. Click "Generate Synthetic Data" in the application header
2. Configure parameters:
3. Click "Generate" to create the data
4. Download CSV if needed for further analysis

### Working with Annotations

1. Load a file with a spectrogram
2. Use the vertical and horizontal cursors to select an area of interest
3. Enter a label and optional comment for your annotation
4. Click "Create Annotation" to save
5. View all annotations in the annotation list
6. Toggle visibility of annotations using the checkboxes
7. Delete annotations as needed
8. Annotations are automatically saved to the database

### Viewing File Information

After loading a file, the statistics panel shows:
- **Metadata Statistics**: Original file metadata
- **Receiver Properties**: Calculated parameters from the signal
- **Transmission Statistics**: Details about detected transmissions (requires AirVIEW)

### Managing Multiple Files

- Use tabs to work with multiple files simultaneously
- Double-click tab titles to rename them
- Use the "Add Tab" button to create new tabs

## Project Structure

```
GC3/
├── backend/                 # Flask server
│   ├── app.py               # Main application
│   ├── Annotation.py        # Signal annotation handling
│   ├── FileData.py          # File data models
│   ├── SigMF.py             # Metadata processing
│   └── airview/             # Signal detection module
├── frontend/                # React app
│   ├── public/              # Static assets
│   └── src/                 # Source code
│       ├── components/      # React components
│       ├── App.tsx          # Main component
│       └── App.css          # Styling
```

## Contributors

- Jun Cho
- Will Cho 
- Grace Johnson
- Connor Whynott
