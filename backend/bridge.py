from flask import Flask, request, jsonify
import subprocess, threading

app = Flask(__name__)
@app.route('http://127.0.0.1:5000/launch-sdr-gui', methods=['POST'])
def launch_sdr_gui():
    data = request.json
    serial = data.get('serial')
    connected = data.get('connected')

    def _launch():
        subprocess.run(['python3', 'bridged-gui.py', serial, str(int(connected))])

    threading.Thread(target=_launch, daemon=True).start()
    return jsonify(status='launched')


if __name__ == '__main__':
    app.run(port=5000)