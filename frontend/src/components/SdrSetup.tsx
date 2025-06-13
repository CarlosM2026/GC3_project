import React, { useState } from 'react';

export interface SdrConfig {
  serial: string;
  type_: string;
  connected: boolean;
}

interface SdrSetupProps {
  onConfig: (config: SdrConfig) => void;
}

const SdrSetup: React.FC<SdrSetupProps> = ({ onConfig }) => {
  const [open, setOpen] = useState(false);
  const [serial, setSerial] = useState('');
  const [type_, setType] = useState('');
  const [connected, setConnected] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    onConfig({ serial, type_, connected });
    setOpen(false);

    await fetch ('http://127.0.0.1:5000/launch-sdr-gui', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ serial, type_, connected })
    })
    console.log("fetch complete")
  };

  return (
    <>
      <button
        className="generate-button"
        onClick={() => setOpen(true)}
      >
        SDR Setup
      </button>

      {open && (
        <div className="dual-popup-overlay">
          <div className="dual-popup-container">
            <div className="generate-popup-section">
              <div className="popup-content">
                <div className="popup-header">
                  <h4>SDR Configuration</h4>
                  <button
                    className="data-close-button"
                    onClick={() => setOpen(false)}
                  >
                    ×
                  </button>
                </div>
                <form onSubmit={handleSubmit}>
                  <label>
                    Serial Number:
                    <input
                      type="text"
                      value={serial}
                      onChange={e => setSerial(e.target.value)}
                      required
                    />
                  </label>
                    <label>
                      <input type="radio" name="myUSRP" value="USRP" onChange={e => setType(e.target.value)}/>
                      USRP
                    </label>
                    <label>
                      <input type="radio" name="myRTL" value="RTL" onChange={e => setType(e.target.value)}/>
                      RTL
                    </label>
                    <label>
                      <input type="radio" name="myHackRF" value="HackRF" onChange={e => setType(e.target.value)}/>
                      HackRF
                    </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={connected}
                      onChange={e => setConnected(e.target.checked)}
                    />
                    SDR Connected?
                  </label>
                  <button type="submit" className="generate-button">
                    Save
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SdrSetup;