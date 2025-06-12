import React from 'react';
import Generate from './Generate';
import SdrSetup, { SdrConfig } from './SdrSetup';

interface ControlPanelProps {
  onPlotGenerated: (plot: string | null) => void;
  onSdrConfig: (config: SdrConfig) => void;
};

const ControlPanel: React.FC<ControlPanelProps> = ({ onPlotGenerated, onSdrConfig }) => (
  <div className="control-buttons">
    <Generate onPlotGenerated={onPlotGenerated} />
    <SdrSetup onConfig={onSdrConfig} />
  </div>
);

export default ControlPanel;