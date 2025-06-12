/** 
 * CS-410: Frontend for uploading files, generating spectrograms, and interacting with MongoDB
 * @file App.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 */

import { useState } from 'react';
import './App.css';
import DisplayTabs from './components/Tabs';
import ControlPanel from './components/ControlPanel';
import { SdrConfig } from './components/SdrSetup';

function App() {
  const [plot, setPlot] = useState<string | null>(null);
  const [sdrConfig, setSdrConfig] = useState<SdrConfig | null>(null);

  const handlePlotGenerated = (plot: string | null) => {
    setPlot(plot);
  };

  const handleSdrConfig = (config: SdrConfig) => {
    setSdrConfig(config);
    console.log('SDR config saved:', config);
    // IT IS SAVED HERE
  };

  const repoUrl = 'https://github.com/DanielJunsangCho/GC3';

  return (
    <main className="enhanced-app-container">
      {/* Application Header */}
      <header className="app-header">
        <div className="header-left">
          <a href={repoUrl} target="_blank" rel="noopener noreferrer" className="github-link">
            <img src="images/github-logo.png" alt="GitHub Repository" className="github-logo" />
          </a>
          <img src="images/GC3-logo.png" alt="GC3 Logo" className="app-logo" />
        </div>

        {/* Combined Generate + SDR controls */}
        <ControlPanel
          onPlotGenerated={handlePlotGenerated}
          onSdrConfig={handleSdrConfig}
        />
      </header>

      {/* Display current SDR config if available */}
      {sdrConfig && (
        <div className="sdr-status">
          <p><strong>SDR Serial:</strong> {sdrConfig.serial}</p>
          <p><strong>Connected:</strong> {sdrConfig.connected ? 'Yes' : 'No'}</p>
        </div>
      )}

      {/* Main Layout: Upload Controls + Metadata + Tabs */}
      <div className="upload-metadata-wrapper">
        <DisplayTabs />
      </div>

      {/* Optionally render the latest plot elsewhere */}
      {plot && (
        <div className="spectrogram-preview">
          <img
            src={`data:image/png;base64,${plot}`}
            alt="Generated Spectrogram"
          />
        </div>
      )}
    </main>
  );
}

export default App;