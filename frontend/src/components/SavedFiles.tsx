/**
 * CS-410: Saved Files library component
 * @file SavedFiles.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 * @description This component is used to display the saved files for the application.
 */

import React, { useState, useEffect } from 'react';
import { IconButton } from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';

// defines interface to be user in FileHandle
interface SavedFile {
  _id: string;
  filename: string;
}

interface SavedFilesProps {
  savedFiles: SavedFile[];
  onDelete: (fileId: string) => void;
  onLoad: (fileId: string) => void;
}

const SavedFiles: React.FC<SavedFilesProps> = ({ savedFiles, onDelete, onLoad }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [files, setFiles] = useState(savedFiles);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  // Update local state when props change
  useEffect(() => {
    setFiles(savedFiles);
  }, [savedFiles]);

  // Filter saved files based on the search term
  const filteredFiles = files.filter((file) =>
    file.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleFilenameChange = async (file: SavedFile, newName: string) => {
    if (!newName.trim() || newName === file.filename) {
      setEditingId(null);
      return;
    }

    const index = files.findIndex(f => f._id === file._id);
    if (index !== -1) {
      try {
        const response = await fetch(`http://127.0.0.1:5000/${file._id}/rename_file`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ filename: newName }),
        });
        
        if (response.ok) {
          setFiles(prevFiles => 
            prevFiles.map(f => 
              f._id === file._id ? { ...f, filename: newName } : f
            )
          );
        } else {
          console.error('Failed to update filename');
        }
      } catch (error) {
        console.error('Error updating filename:', error);
      }
    }
    setEditingId(null);
  }

  return (
    <div className="saved-files">
      {/* Fixed Header */}
      <div className="saved-files-header">
        <h2>Saved Files</h2>
      </div>
      
      {/* Fixed Search Bar */}
      <div className="search-container">
        <input
          type="text"
          placeholder="Search files..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-bar"
        />
      </div>
      
      {/* Scrollable File List */}
      <ul>
        {filteredFiles.length > 0 ? (
          filteredFiles.map((file) => (
            <li key={file._id} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={() => onDelete(file._id)}
                className="delete-btn"
                title="Delete file"
              >
                <img src="/images/trash-icon.png" alt="Delete" className="trash-icon" />
              </button>
              {editingId === file._id ? (
                <input
                  type="text"
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onBlur={() => handleFilenameChange(file, editingName)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleFilenameChange(file, editingName);
                    } else if (e.key === 'Escape') {
                      setEditingId(null);
                    }
                  }}
                  autoFocus
                  style={{ flex: 1 }}
                />
              ) : (
                <span className="file-name" style={{ flex: 1 }}>{file.filename}</span>
              )}
              <IconButton
                size="small"
                onClick={() => {
                  setEditingId(file._id);
                  setEditingName(file.filename);
                }}
                sx={{
                  padding: '4px',
                  marginRight: '8px',
                  backgroundColor: 'transparent !important',
                  color: 'black',
                  '&:hover': { backgroundColor: 'transparent' },
                }}
              >
                <EditIcon fontSize="small" sx={{ color: 'black' }} />
              </IconButton>
              <button onClick={() => onLoad(file._id)}>Load</button>
            </li>
          ))
        ) : (
          <li>No files match your search.</li>
        )}
      </ul>
    </div>
  );
};

export default SavedFiles;