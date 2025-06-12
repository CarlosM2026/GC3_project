/**
 * CS-410: Tab component
 * @file Tabs.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 * @description This component is used to display the tabs for the application.
 */

import React, { useState } from 'react';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import { IconButton, TextField } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import FileHandle from './FileHandle';
import Statistics from './Statistics';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function CustomTabPanel(props: TabPanelProps) {
  const { children, value, index } = props;
  return (
    <div role="tabpanel" hidden={value !== index} id={`tabpanel-${index}`} aria-labelledby={`tab-${index}`}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const DisplayTabs: React.FC = () => {
  const [value, setValue] = useState(0);
  const [tabs, setTabs] = useState<{ id: number; label: string; fileId: string | null; airview_annotations?: any[] }[]>([
    { id: 0, label: 'Tab 1', fileId: null, airview_annotations: [] },
  ]);  
  const [editingTabId, setEditingTabId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState<string>('');
  

  const handleChange = (_: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleAdd = () => {
    const newId = tabs.length > 0 ? Math.max(...tabs.map(tab => tab.id)) + 1 : 0;
    setTabs([...tabs, { id: newId, label: `Tab ${tabs.length + 1}`, fileId: null }]);
  };

  const handleDelete = (indexToDelete: number) => {
    setTabs(prevTabs => prevTabs.filter((_, index) => index !== indexToDelete));
    if (indexToDelete <= value) {
      setValue(prev => Math.max(0, prev - 1));
    }
  };

  const updateFileId = (tabId: number, newFileId: string | null, airview_annotations?: any[]) => {
    setTabs(prevTabs =>
      prevTabs.map(tab =>
        tab.id === tabId ? { ...tab, fileId: newFileId, airview_annotations } : tab
      )
    );
  };  

  const handleDoubleClick = (tabId: number, currentLabel: string) => {
    setEditingTabId(tabId);
    setEditValue(currentLabel);
  };

  const handleRenameSubmit = (tabId: number) => {
    setTabs(prevTabs =>
      prevTabs.map(tab =>
        tab.id === tabId ? { ...tab, label: editValue } : tab
      )
    );
    setEditingTabId(null);
  };

  const handleKeyDown = (event: React.KeyboardEvent, tabId: number) => {
    if (event.key === 'Enter') {
      handleRenameSubmit(tabId);
    } else if (event.key === 'Escape') {
      setEditingTabId(null);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={value} onChange={handleChange} variant="scrollable" scrollButtons="auto" sx={{height:"30px"}}>
          {tabs.map((tab, index) => (
            <Tab
              key={index}
              label={
                editingTabId === tab.id ? (
                  <TextField
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, tab.id)}
                    onBlur={() => handleRenameSubmit(tab.id)}
                    autoFocus
                    size="small"
                    variant="standard"
                    sx={{
                      width: '100px',
                      '& .MuiInputBase-root': {
                        padding: '0',
                        margin: '0',
                        height: '30px',
                        '& input': {
                          padding: '6px 0',
                          fontSize: '0.875rem',
                        },
                      },
                      '& .MuiInput-underline:before': {
                        borderBottom: 'none',
                      },
                      '& .MuiInput-underline:after': {
                        borderBottom: 'none',
                      },
                      '& .MuiInput-underline:hover:not(.Mui-disabled):before': {
                        borderBottom: 'none',
                      },
                    }}
                  />
                ) : (
                  <span onDoubleClick={() => handleDoubleClick(tab.id, tab.label)}>
                    {tab.label}
                  </span>
                )
              }
              icon={
                <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleDelete(index); }}>
                  <CloseIcon fontSize="small" />
                </IconButton>
              }
              iconPosition="end"
              sx={{ textTransform: 'none' }}
            />
          ))}
          <Button variant="contained" onClick={handleAdd}>Add Tab</Button>
        </Tabs>
      </Box>

      {tabs.map((tab, index) => (
        <CustomTabPanel key={index} value={value} index={index}>
          <div className="tab-content">
            {/* Conditionally render Statistics only if fileId exists */}
            {tab.fileId ? (
              <div className="statistics-container">
                <Statistics fileId={tab.fileId} airview_annotations={tab.airview_annotations ?? []} />
              </div>
            ) : (
              <div className="statistics-container">
                <p>No file selected. Please select a file to view statistics.</p>
              </div>
            )}
            <div className="file-handle-container">
              <FileHandle
                fileId={tab.fileId}
                onFileSelect={(fileId, airview_annotations) => updateFileId(tab.id, fileId, airview_annotations)}
              />
            </div>
          </div>
        </CustomTabPanel>
      ))}
    </Box>
  );
}

export default DisplayTabs;