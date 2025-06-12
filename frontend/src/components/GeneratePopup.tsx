/**
 * CS-410: Synthetic Data Popup component.
 * @file GeneratePopup.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 * @description This component is used to display synthetic data for the spectrogram.
 */

import React, { useState } from 'react';

interface GeneratePopupProps {
  onClose: () => void;
  onGenerate: (data: {
    rows: number | null;
    cols: number | null;
    numTransmitters: number | null;
    transmitterMean: number | null;
    transmitterSd: number | null;
    noiseMean: number | null;
    noiseSd: number | null;
    bandwidth: number | null;
    activeTime: number | null;
    matrixFilename: string;
    transmittersFilename: string;
    placementMethod: string;
  }) => void;
}

const GeneratePopup: React.FC<GeneratePopupProps> = ({ onClose, onGenerate }) => {
  const [rows, setRows] = useState<number | null>(1000);
  const [cols, setCols] = useState<number | null>(1024);
  const [numTransmitters, setNumTransmitters] = useState<number | null>(5);
  const [transmitterMean, setTransmitterMean] = useState<number | null>(-75);
  const [transmitterSd, setTransmitterSd] = useState<number | null>(2);
  const [noiseMean, setNoiseMean] = useState<number | null>(-109);
  const [noiseSd, setNoiseSd] = useState<number | null>(10);
  const [bandwidth, setBandwidth] = useState<number | null>(200);
  const [activeTime, setActiveTime] = useState<number | null>(10);
  const [placementMethod, setPlacementMethod] = useState<string>('equally_spaced');
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (
    value: string, 
    setter: React.Dispatch<React.SetStateAction<number | null>>
  ) => {
    if (value === '') {
      setter(null);
    } else {
      setter(Number(value));
    }
    // Clear error when user starts typing
    setError(null);
  };

  const validateForm = () => {
    // Only set the first error encountered
    if (rows === null) {
      setError('Rows is required');
      return false;
    }
    if (cols === null) {
      setError('Columns is required');
      return false;
    }
    if (numTransmitters === null) {
      setError('Number of Transmitters is required');
      return false;
    }
    if (transmitterMean === null) {
      setError('Transmitter Mean is required');
      return false;
    }
    if (transmitterSd === null) {
      setError('Transmitter Standard Deviation is required');
      return false;
    }
    if (noiseMean === null) {
      setError('Noise Mean is required');
      return false;
    }
    if (noiseSd === null) {
      setError('Noise Standard Deviation is required');
      return false;
    }
    if (bandwidth === null) {
      setError('Bandwidth is required');
      return false;
    }
    if (activeTime === null) {
      setError('Active Time is required');
      return false;
    }
    
    return true;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return; // Stop if validation fails
    }
    
    onGenerate({
      rows,
      cols,
      numTransmitters,
      transmitterMean,
      transmitterSd,
      noiseMean,
      noiseSd,
      bandwidth,
      activeTime,
      matrixFilename: 'output_matrix.csv',
      transmittersFilename: 'output_transmitters.csv',
      placementMethod,
    });
  };

  return (
    <div className="popup-content">
      <div className="popup-header">
        <h4>Generate Data</h4>
        <button className="data-close-button" onClick={onClose}>×</button>
      </div>
      <form onSubmit={handleSubmit}>
        {error && (
          <div className="single-error-message">{error}</div>
        )}
        
        <fieldset>
          <legend>Placement Method</legend>
          <label>
            <input
              type="radio"
              value="equally_spaced"
              checked={placementMethod === 'equally_spaced'}
              onChange={(e) => setPlacementMethod(e.target.value)}
            />
            Equally Spaced
          </label>
          <label>
            <input
              type="radio"
              value="random"
              checked={placementMethod === 'random'}
              onChange={(e) => setPlacementMethod(e.target.value)}
            />
            Random
          </label>
        </fieldset>
        
        <label>
          Rows:
          <input 
            type="number" 
            value={rows === null ? '' : rows} 
            onChange={(e) => handleInputChange(e.target.value, setRows)} 
          />
        </label>
        
        <label>
          Columns:
          <input 
            type="number" 
            value={cols === null ? '' : cols} 
            onChange={(e) => handleInputChange(e.target.value, setCols)} 
          />
        </label>
        
        <label>
          Number of Transmitters:
          <input 
            type="number" 
            value={numTransmitters === null ? '' : numTransmitters} 
            onChange={(e) => handleInputChange(e.target.value, setNumTransmitters)} 
          />
        </label>
        
        <label>
          Transmitter Mean:
          <input 
            type="number" 
            value={transmitterMean === null ? '' : transmitterMean} 
            onChange={(e) => handleInputChange(e.target.value, setTransmitterMean)} 
          />
        </label>
        
        <label>
          Transmitter Standard Deviation:
          <input 
            type="number" 
            value={transmitterSd === null ? '' : transmitterSd} 
            onChange={(e) => handleInputChange(e.target.value, setTransmitterSd)} 
          />
        </label>
        
        <label>
          Noise Mean:
          <input 
            type="number" 
            value={noiseMean === null ? '' : noiseMean} 
            onChange={(e) => handleInputChange(e.target.value, setNoiseMean)} 
          />
        </label>
        
        <label>
          Noise Standard Deviation:
          <input 
            type="number" 
            value={noiseSd === null ? '' : noiseSd} 
            onChange={(e) => handleInputChange(e.target.value, setNoiseSd)} 
          />
        </label>
        
        <label>
          Bandwidth:
          <input 
            type="number" 
            value={bandwidth === null ? '' : bandwidth} 
            onChange={(e) => handleInputChange(e.target.value, setBandwidth)} 
          />
        </label>
        
        <label>
          Active Time:
          <input 
            type="number" 
            value={activeTime === null ? '' : activeTime} 
            onChange={(e) => handleInputChange(e.target.value, setActiveTime)} 
          />
        </label>
        
        <button type="submit">Generate</button>
      </form>
    </div>
  );
};

export default GeneratePopup;