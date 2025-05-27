"use client"

import React, { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Download, CheckCircle, Loader2, Trash2, FileText, Upload } from 'lucide-react';
import { ModelInfo } from '../types/model';

interface SidebarProps {
  theme: 'light' | 'dark';
  models: ModelInfo[];
  loadingModels: boolean;
  downloadingModel: string | null;
  settingModel: string | null;
  isModelActive: (model: ModelInfo) => boolean;
  handleDownload: (modelId: string) => void;
  handleSetActive: (modelId: string) => void;
}

export default function Sidebar({
  theme,
  models,
  loadingModels,
  downloadingModel,
  settingModel,
  isModelActive,
  handleDownload,
  handleSetActive
}: SidebarProps) {
  return (
    <div className={`h-full flex flex-col p-4 border-l ${
      theme === 'dark' ? 'bg-gray-900 border-gray-800 text-gray-200' : 'bg-gray-100 border-gray-200 text-gray-800'
    } w-64 flex-shrink-0`}>
      <div className={`flex-1 overflow-y-auto`}>
        <h3 className="text-sm font-semibold mb-2 sticky top-0 backdrop-blur-sm pt-4 px-4 -mx-4">Model Management</h3>
        {loadingModels ? (
          <div className='text-xs text-gray-500 p-2'>Loading models...</div>
        ) : models.length === 0 ? (
          <div className='text-xs text-gray-500 p-2'>No models found.</div>
        ) : (
          <ul className="space-y-2 px-4 -mx-4 pb-4">
            {models.map(model => (
              <li key={model.id} className={`p-2 rounded border ${theme === 'dark' ? 'border-gray-700 bg-gray-800' : 'border-gray-300 bg-white'}`}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium truncate mr-2" title={model.description || model.name}>{model.name}</span>
                  {isModelActive(model) ? (
                    <span className="flex items-center text-xs text-green-500"><CheckCircle size={14} className="mr-1"/> Active</span>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleSetActive(model.id)}
                      disabled={settingModel === model.id || downloadingModel !== null || model.status !== 'available'}
                      className="text-xs h-7"
                    >
                      {settingModel === model.id ? <Loader2 size={12} className="mr-1 animate-spin"/> : null}
                      Set Active
                    </Button>
                  )}
                </div>
                <div className="text-xs text-gray-500 mb-2">
                  ID: {model.id} {model.size_gb ? `| Size: ${model.size_gb.toFixed(1)} GB` : ''}
                </div>
                {model.status !== 'available' && (
                    <div className="flex justify-end">
                        <Button
                            size="sm"
                            variant="default"
                            onClick={() => handleDownload(model.id)}
                            disabled={downloadingModel === model.id || settingModel !== null}
                            className="text-xs h-7 bg-blue-600 hover:bg-blue-700"
                        >
                            {downloadingModel === model.id ? (
                                <Loader2 size={12} className="mr-1 animate-spin"/>
                            ) : (
                                <Download size={12} className="mr-1" />
                            )}
                            Download
                        </Button>
                    </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
