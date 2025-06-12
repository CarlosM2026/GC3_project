/**
 * CS-410: Component for annotation features
 * @file Annotations.tsx
 * @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 * @collaborators None
 * @description This component is used to upload, view and manage files.
 */

import React, { useState } from 'react';

export interface Annotation {
  id: string;
  corners: { freq1: number; time1: number; freq2: number; time2: number };
  label: string;
  comment: string;
  display: boolean;
}

interface AnnotationsProps {
    annotations: Annotation[];
    annotationLabel: string;
    annotationComment: string;
    horizontalCursors: number[];
    verticalCursors: number[];
    calculateHorizontalCursorPosition: (position: number) => number;
    calculateVerticalCursorPosition: (position: number) => number;
    setAnnotationLabel: (label: string) => void;
    setAnnotationComment: (comment: string) => void;
    createAnnotation: () => void;
    deleteAnnotation: (annotationId: string) => void;
    toggleAnnotationDisplay: (annotationId: string, isChecked: boolean) => void;
    showCursors: boolean; // Add showCursors
    setShowCursors: (value: boolean) => void; // Add setShowCursors
  }

const formatFrequency = (frequency: number | null): string => {
    if (frequency === null) return 'N/A';
    if (frequency >= 1000000) {
      return `${(frequency / 1000000).toFixed(2)} MHz`;
    } else if (frequency >= 1000) {
      return `${(frequency / 1000).toFixed(2)} kHz`;
    } else {
      return `${frequency.toFixed(2)} Hz`;
    }
  };

const Annotations: React.FC<AnnotationsProps> = ({
  annotations,
  annotationLabel,
  annotationComment,
  horizontalCursors,
  verticalCursors,
  calculateHorizontalCursorPosition,
  calculateVerticalCursorPosition,
  setAnnotationLabel,
  setAnnotationComment,
  createAnnotation,
  deleteAnnotation,
  toggleAnnotationDisplay,
  showCursors,
  setShowCursors,
}) => {
    const [expandedAnnotationId, setExpandedAnnotationId] = useState<string | null>(null);

    const toggleExpand = (annotationId: string) => {
    setExpandedAnnotationId((prev) => (prev === annotationId ? null : annotationId));
  };
  return (
    <div className="annotations-container">
        <h3>Annotations</h3>
        <div className="toggle-cursors">
            <label className="toggle-switch">
                <input
                type="checkbox"
                checked={showCursors}
                onChange={(e) => setShowCursors(e.target.checked)}
                />
                <span className="slider"></span>
            </label>
            <span>{showCursors ? "Cursors On" : "Cursors Off"}</span>
        </div>

        {/* Annotation Creation Section */}
        <div className="annotation-creation-box">
            {/* Red-bordered rectangle with corner labels */}
            <div className="annotation-preview">
                <div className="corner-label top-left">
                    ({formatFrequency(verticalCursors.length > 0 ? calculateVerticalCursorPosition(verticalCursors[0]) : null)}, 
                    {horizontalCursors.length > 0 ? calculateHorizontalCursorPosition(horizontalCursors[0]).toFixed(2) : 'N/A'})
                </div>
                <div className="corner-label top-right">
                    ({formatFrequency(verticalCursors.length > 0 ? calculateVerticalCursorPosition(verticalCursors[1]) : null)}, 
                    {horizontalCursors.length > 0 ? calculateHorizontalCursorPosition(horizontalCursors[0]).toFixed(2) : 'N/A'})
                </div>
                <div className="corner-label bottom-left">
                    ({formatFrequency(verticalCursors.length > 0 ? calculateVerticalCursorPosition(verticalCursors[0]) : null)}, 
                    {horizontalCursors.length > 0 ? calculateHorizontalCursorPosition(horizontalCursors[1]).toFixed(2) : 'N/A'})
                </div>
                <div className="corner-label bottom-right">
                    ({formatFrequency(verticalCursors.length > 0 ? calculateVerticalCursorPosition(verticalCursors[1]) : null)}, 
                    {horizontalCursors.length > 0 ? calculateHorizontalCursorPosition(horizontalCursors[1]).toFixed(2) : 'N/A'})
                </div>
            </div>

            {/* Inputs and Create Annotation Button */}
            <div className="annotation-inputs">
                <div className="inputs-container">
                <input
                    type="text"
                    placeholder="Enter annotation label"
                    value={annotationLabel}
                    onChange={(e) => setAnnotationLabel(e.target.value)}
                    className="annotation-label-input"
                />
                <textarea
                    placeholder="Enter optional comment"
                    value={annotationComment}
                    onChange={(e) => setAnnotationComment(e.target.value)}
                    className="annotation-comment-input"
                ></textarea>
                </div>
                <button
                className="btn create-annotation-btn"
                onClick={createAnnotation}
                disabled={!annotationLabel.trim()}
                >
                Create Annotation
                </button>
            </div>
            </div>

        {/* Saved Annotations Section */}
        <div className="saved-annotations-container">
            <h2>Saved Annotations</h2>
            <table className="annotations-table">
                <thead>
                <tr>
                    <th></th> {/* No header for the delete column */}
                    <th>Label</th>
                    <th>Start Time (s)</th>
                    <th>End Time (s)</th>
                    <th>Start Freq (Hz)</th>
                    <th>End Freq (Hz)</th>
                    <th>Display</th>
                    <th>Comment</th> {/* New column for the expand/collapse button */}
                </tr>
                </thead>
                <tbody>
                {annotations.map((annotation) => (
                    <React.Fragment key={annotation.id}>
                    <tr>
                        <td>
                        <button
                            className="btn delete-annotation-btn"
                            onClick={() => deleteAnnotation(annotation.id)}
                        >
                            <img src="images/trash-icon.png" alt="Delete" className="trash-icon" />
                        </button>
                        </td>
                        <td>{annotation.label}</td>
                        <td>{annotation.corners.time1.toFixed(2)}</td>
                        <td>{annotation.corners.time2.toFixed(2)}</td>
                        <td>{annotation.corners.freq1.toFixed(2)}</td>
                        <td>{annotation.corners.freq2.toFixed(2)}</td>
                        <td>
                        <input
                            type="checkbox"
                            checked={annotation.display}
                            onChange={(e) => toggleAnnotationDisplay(annotation.id, e.target.checked)}
                        />
                        </td>
                        <td>
                        <button
                            className="expand-comment-btn"
                            onClick={() => toggleExpand(annotation.id)}
                        >
                            {expandedAnnotationId === annotation.id ? '▲' : '▼'}
                        </button>
                        </td>
                    </tr>
                    {expandedAnnotationId === annotation.id && (
                        <tr className="annotation-comment-row">
                        <td colSpan={8} className="annotation-comment-cell">
                            <strong>Comment:</strong> {annotation.comment || 'No comment provided'}
                        </td>
                        </tr>
                    )}
                    </React.Fragment>
                ))}
                </tbody>
            </table>
        </div>
        </div>
  );
};

export default Annotations;