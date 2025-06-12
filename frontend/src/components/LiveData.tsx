// src/components/SdrSetup.tsx
import React, { useState } from 'react';

export interface SdrConfig {
  serial: string;
  connected: boolean;
}

interface SdrSetupProps {
  onConfig: (config: SdrConfig) => void;
}

const SdrSetup: React.FC<SdrSetupProps> = ({ onConfig }) => {
  const [open, setOpen] = useState(false);
  const [serial, setSerial] = useState('');
  const [connected, setConnected] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfig({ serial, connected });
    setOpen(false);
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

  