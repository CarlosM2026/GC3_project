/**
 * CS-410: Generate Synthetic data component
 * @file Generate.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 * @description This component is used to generate synthetic data for the spectrogram.
 */

import React, { useState } from 'react';
import GeneratePopup from './GeneratePopup';

const Generate: React.FC<{ onPlotGenerated: (plot: string | null) => void }> = ({ onPlotGenerated }) => {
  const [showPopup, setShowPopup] = useState(false);
  const [generatedPlot, setGeneratedPlot] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<string | null>(null);
  const [filename, setFilename] = useState<string>("generated_data.csv");

  const handleGenerateClick = () => {
    setShowPopup(true);
  };

  const handleClosePopup = () => {
    setShowPopup(false);
    setGeneratedPlot(null);
    setCsvData(null);
  };

  const handleClosePlot = () => {
    setGeneratedPlot(null);
    setCsvData(null);
  };

  const handleDownloadCSV = () => {
    if (!csvData) return;
    
    // Create a blob from the CSV data
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    
    // Create a URL for the blob
    const url = URL.createObjectURL(blob);
    
    // Create a temporary link element
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    
    // Append the link to the body
    document.body.appendChild(link);
    
    // Trigger a click on the link
    link.click();
    
    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleGenerateData = async (data: any) => {
    try {
      setFilename(`${data.placementMethod}_transmitters_${data.numTransmitters}_${Date.now()}.csv`);
      
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setGeneratedPlot(result.plot);
        setCsvData(result.csv);
        onPlotGenerated(result.plot);
      } else {
        const error = await response.json();
        console.error('Backend error:', error.error);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <>
      <button className="generate-button" onClick={handleGenerateClick}>
        Generate Synthetic Data
      </button>
      
      {showPopup && (
        <div className="dual-popup-overlay">
          <div className="dual-popup-container">
            <div className="generate-popup-section">
              <GeneratePopup 
                onClose={handleClosePopup} 
                onGenerate={handleGenerateData} 
              />
            </div>
            
            {generatedPlot && (
              <div className="spectrogram-popup-section">
                <div className="spectrogram-popup-content">
                  <div className="spectrogram-header">
                    <h4>Generated Spectrogram</h4>
                    <button className="spec-close-button" onClick={handleClosePlot}>×</button>
                  </div>
                  <div className="spectrogram-image-container">
                    <img 
                      src={`data:image/png;base64,${generatedPlot}`} 
                      alt="Generated Spectrogram" 
                      className="spectrogram-image"
                    />
                  </div>
                  
                  {csvData && (
                    <div className="download-container">
                      <button 
                        className="download-btn"
                        onClick={handleDownloadCSV}
                      >
                        Download CSV Data
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default Generate;