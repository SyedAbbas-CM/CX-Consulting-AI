"use client"

import React, { useState, useEffect, useCallback } from 'react';
import { useTheme } from 'next-themes'; // Import useTheme

// Import Model Management related things
import {
  getLlmConfig,
  downloadModel,
  setActiveModel,
  listModels,
  listVectorDbs,
  downloadVectorDb,
  getLlmRuntimeStatus,
  getModelStatus,
} from '@/lib/apiClient';
import { LlmConfigResponse } from '@/types/config';
import { ModelInfo } from '@/types/model';
import type { VectorDbInfo } from '@/lib/apiClient';
import { Button } from '@/components/ui/button';
import { Download, CheckCircle, Loader2, LogOut, Sun, Moon, Settings, User, ChevronsUpDown } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast'; // Import toast
import Link from 'next/link';
import { useAuth } from '@/context/auth-context'; // Import auth context
import CollapsibleProjectSidebar from '@/components/collapsible-project-sidebar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();
  const { user, logout } = useAuth();

  // Add mounted state
  const [mounted, setMounted] = useState(false);

  // --- State from Sidebar for Model Management ---
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModelPath, setActiveModelPath] = useState<string>('Unknown');
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [settingModel, setSettingModel] = useState<string | null>(null);
  const [llmConfig, setLlmConfig] = useState<LlmConfigResponse | null>(null);
  const [isLoadingConfig, setIsLoadingConfig] = useState<boolean>(true);

  // Runtime load status (whether model actually resident in memory)
  const [llmLoaded, setLlmLoaded] = useState<boolean>(false);
  const [llmLoadError, setLlmLoadError] = useState<string | null>(null);

  // Vector DB management state
  const [vectorDbs, setVectorDbs] = useState<VectorDbInfo[]>([]);
  const [vectorDbActiveId, setVectorDbActiveId] = useState<string | null>(null);
  const [loadingVectorDbs, setLoadingVectorDbs] = useState<boolean>(true);
  const [downloadingVectorDb, setDownloadingVectorDb] = useState<string | null>(null);

  // Progress state
  const [modelProgress, setModelProgress] = useState<Record<string, number>>({});

  // --- Effects from Sidebar for Model Management ---
  const fetchModelsAndConfig = useCallback(async () => {
    setLoadingModels(true);
    setIsLoadingConfig(true);
    try {
      // Only fetch if logged in (token likely needed)
      if (typeof window !== 'undefined' && localStorage.getItem("authToken")) {
        const modelData = await listModels();
        setModels(modelData.available_models || []);
        setActiveModelPath(modelData.active_model_path || 'Unknown');

        const configData = await getLlmConfig();
        setLlmConfig(configData);

        // Runtime status
        try {
          const rtStatus = await getLlmRuntimeStatus();
          setLlmLoaded(rtStatus.loaded);
          setLlmLoadError(rtStatus.load_error || null);
        } catch (e) {
          console.warn('Failed to fetch LLM runtime status', e);
        }

        // Fetch vector dbs
        const vectorData = await listVectorDbs();
        setVectorDbs(vectorData.available_vector_dbs || []);
        setVectorDbActiveId(vectorData.active_db_id || null);
      } else {
        console.warn("DashboardLayout: Not fetching models/config, no auth token found.");
        // Handle state if not logged in? Maybe clear models/config?
        setModels([]);
        setLlmConfig(null);
        setActiveModelPath('Unknown');
        setVectorDbs([]);
        setVectorDbActiveId(null);
      }
    } catch (error) {
      console.error("DashboardLayout: Error fetching models or config:", error);
      toast({ variant: "destructive", title: "Load Failed", description: "Could not load model or configuration data." });
      setModels([]);
      setLlmConfig(null); // Ensure config is null on error
    } finally {
      setLoadingModels(false);
      setIsLoadingConfig(false);
      setLoadingVectorDbs(false);
    }
  }, [toast, user]);

  useEffect(() => {
    fetchModelsAndConfig();
  }, [fetchModelsAndConfig]);

  // Effect to set mounted state
  useEffect(() => {
    setMounted(true);
  }, []);

  // Effect: poll LLM runtime status every 3 s until loaded or error
  useEffect(() => {
    // Only start polling after initial mount & when user has auth token
    if (typeof window === 'undefined') return;
    if (!localStorage.getItem('authToken')) return;

    const interval = setInterval(async () => {
      try {
        const rt = await getLlmRuntimeStatus();
        setLlmLoaded(rt.loaded);
        setLlmLoadError(rt.load_error || null);
        // Stop polling once model is loaded or permanent error captured
        if (rt.loaded || rt.load_error) {
          clearInterval(interval);
        }
      } catch {
        /* swallow */
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // --- Handlers from Sidebar for Model Management ---
  const pollModelProgress = useCallback((modelId: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await getModelStatus(modelId);
        if (status.status === "available") {
          setModelProgress(prev => ({ ...prev, [modelId]: 100 }));
          clearInterval(interval);
          fetchModelsAndConfig();
        } else if (status.download_progress !== undefined && status.download_progress !== null) {
          const pct = Number(status.download_progress ?? 0);
          setModelProgress(prev => ({ ...prev, [modelId]: pct }));
        }
      } catch (e) {
        console.warn("progress poll failed", e);
      }
    }, 4000);
  }, [fetchModelsAndConfig]);

  const handleDownload = async (modelId: string) => {
    setDownloadingModel(modelId);
    setModelProgress(prev => ({ ...prev, [modelId]: 0 }));
    try {
      const result = await downloadModel(modelId);
      toast({ title: "Download Started", description: result.message });
      pollModelProgress(modelId);
    } catch (error: any) {
      console.error("Error downloading model:", error);
      const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
      toast({ variant: "destructive", title: "Download Failed", description: `Failed to start model download: ${errorMessage}` });
    } finally {
      setDownloadingModel(null);
    }
  };

  const handleSetActive = async (modelId: string) => {
    setSettingModel(modelId);
    try {
      const data = await setActiveModel(modelId);
      // alert(data.message);
      toast({ title: "Model Set Active", description: data.message });
      // Refetch config to confirm the change
      const configData = await getLlmConfig();
      setLlmConfig(configData);
      // Update active model path optimistically (or based on configData if reliable)
      const modelInfo = models.find(m => m.id === modelId);
      const newPath = modelInfo?.path || 'Unknown'; // Use actual path if available
      setActiveModelPath(newPath); // Use path from ModelInfo if available
    } catch (error: any) {
      console.error("Error setting active model:", error);
      const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
      // alert(`Failed to set active model: ${errorMessage}`);
      toast({ variant: "destructive", title: "Activation Failed", description: `Failed to set active model: ${errorMessage}` });
    } finally {
      setSettingModel(null);
    }
  };

  // --- Helper from Sidebar for Model Management ---
  const isModelActive = (model: ModelInfo): boolean => {
    // Use llmConfig.model_id if available, otherwise compare paths
    if (llmConfig?.model_id) {
      return llmConfig.model_id === model.id;
    }
    if (!activeModelPath || activeModelPath === 'Unknown' || !model.path) return false;
    // Normalize paths slightly (optional, depends on backend consistency)
    const normalizedActivePath = activeModelPath.replace(/^models\//, '');
    const normalizedModelPath = model.path.replace(/^models\//, '');
    return normalizedModelPath === normalizedActivePath;
  };

  // Vector DB helpers
  const isVectorDbActive = (db: VectorDbInfo): boolean => db.id === vectorDbActiveId;

  const handleDownloadVectorDb = async (dbId: string) => {
    setDownloadingVectorDb(dbId);
    try {
      const result = await downloadVectorDb(dbId);
      toast({ title: "Vector DB Download", description: result.message });
      // Refresh vector-db list shortly after extraction
      setTimeout(() => fetchModelsAndConfig(), 3000);
    } catch (err: any) {
      const msg = err?.errorData?.detail || err?.message || 'Unknown error';
      toast({ variant: "destructive", title: "Vector DB Download Failed", description: msg });
    } finally {
      setDownloadingVectorDb(null);
    }
  };

  // Prevent rendering theme-dependent parts until mounted
  if (!mounted) {
    // Render null or a basic loading state/skeleton on the server and initial client render
    return null;
  }

  return (
    <div className={`flex h-screen w-screen overflow-hidden ${
      theme === 'dark' ? 'bg-[#0f1117] text-gray-200' : 'bg-gray-50 text-gray-800'
    }`}>
      {/* Render the new collapsible sidebar only when mounted */}
      <CollapsibleProjectSidebar theme={theme === 'light' ? 'light' : 'dark'} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header Section */}
        <header className={`flex items-center justify-between p-3 border-b ${theme === 'dark' ? 'border-gray-800 bg-[#0f1117]' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex items-center space-x-4 flex-wrap">
            <span className="font-semibold text-lg">CX Consulting AI</span>

            {/* Model Management Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                {/* Current Model Status Display - Now the Trigger */}
                <Button variant="ghost" className="flex items-center space-x-2 text-xs px-2 py-1 h-auto border rounded-md ${theme === 'dark' ? 'border-gray-700 hover:bg-gray-800' : 'border-gray-300 hover:bg-gray-100'}">
                   {isLoadingConfig ? (
                     <Loader2 size={14} className="animate-spin text-gray-500" />
                   ) : llmLoaded ? (
                     <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
                   ) : (
                     <span className='text-red-500 font-bold'>!</span>
                   )}
                   <span className='font-mono truncate max-w-[200px]'>
                      {isLoadingConfig ? 'Loading...' : (llmLoaded ? (llmConfig?.model_id || 'Loaded Model') : 'No Model Loaded')}
                   </span>
                   <ChevronsUpDown size={14} className="text-gray-500 flex-shrink-0" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-72 max-h-96 overflow-y-auto">
                <DropdownMenuLabel>Model Management</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {/* Model List */}
                {loadingModels ? (
                  <DropdownMenuItem disabled>Loading models...</DropdownMenuItem>
                ) : models.length === 0 ? (
                  <DropdownMenuItem disabled>No models found.</DropdownMenuItem>
                ) : (
                  models.map(model => (
                    <DropdownMenuItem
                      key={model.id}
                      className="flex flex-col items-start p-2 !cursor-default" // Allow complex content
                      onSelect={(e) => e.preventDefault()} // Prevent closing on item interaction
                    >
                      <div className="w-full flex justify-between items-center mb-1">
                        <span className="text-sm font-medium truncate mr-2" title={model.description || model.name}>{model.name}</span>
                        {isModelActive(model) ? (
                          <span className="flex items-center text-xs text-green-500"><CheckCircle size={14} className="mr-1"/> Active</span>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => { e.stopPropagation(); handleSetActive(model.id); }} // Stop propagation
                            disabled={settingModel === model.id || downloadingModel !== null || model.status !== 'available'}
                            className="text-xs h-7"
                          >
                            {settingModel === model.id ? <Loader2 size={12} className="mr-1 animate-spin"/> : null}
                            Set Active
                          </Button>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mb-2 w-full">
                        ID: {model.id} {model.size_gb ? `| Size: ${model.size_gb.toFixed(1)} GB` : ''}
                      </div>
                      {model.status !== 'available' && (
                        <div className="w-full flex flex-col space-y-1">
                          {modelProgress[model.id] !== undefined && (
                            <div className="w-full bg-gray-200 rounded h-1 overflow-hidden">
                              <div className="bg-blue-600 h-1" style={{ width: `${modelProgress[model.id].toFixed(0)}%` }} />
                            </div>
                          )}
                          <div className="w-full flex justify-end">
                            <Button
                                size="sm"
                                variant="default"
                                onClick={(e) => { e.stopPropagation(); handleDownload(model.id); }} // Stop propagation
                                disabled={downloadingModel === model.id || settingModel !== null}
                                className="text-xs h-7 bg-blue-600 hover:bg-blue-700"
                            >
                                {downloadingModel === model.id ? (
                                    <Loader2 size={12} className="mr-1 animate-spin" />
                                ) : (
                                    <Download size={12} className="mr-1" />
                                )}
                                {modelProgress[model.id] !== undefined ? `${modelProgress[model.id].toFixed(0)}%` : 'Download'}
                            </Button>
                          </div>
                        </div>
                      )}
                    </DropdownMenuItem>
                  ))
                )}
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Vector DB</DropdownMenuLabel>
                {loadingVectorDbs ? (
                  <DropdownMenuItem disabled>Loading vector DBs...</DropdownMenuItem>
                ) : vectorDbs.length === 0 ? (
                  <DropdownMenuItem disabled>No vector DB defined.</DropdownMenuItem>
                ) : (
                  vectorDbs.map(db => (
                    <DropdownMenuItem
                      key={db.id}
                      className="flex flex-col items-start p-2 !cursor-default"
                      onSelect={e => e.preventDefault()}
                    >
                      <div className="w-full flex justify-between items-center mb-1">
                        <span className="text-sm font-medium truncate mr-2" title={db.description || db.filename}>{db.id}</span>
                        {isVectorDbActive(db) ? (
                          <span className="flex items-center text-xs text-green-500"><CheckCircle size={14} className="mr-1"/> Active</span>
                        ) : null}
                      </div>
                      <div className="text-xs text-gray-500 mb-2 w-full">
                        {db.filename} {db.size_gb ? `| ${db.size_gb.toFixed(1)} GB` : ''}
                      </div>
                      {db.status !== 'available' && (
                        <div className="w-full flex justify-end">
                          <Button
                            size="sm"
                            variant="default"
                            onClick={e => { e.stopPropagation(); handleDownloadVectorDb(db.id); }}
                            disabled={downloadingVectorDb === db.id}
                            className="text-xs h-7 bg-blue-600 hover:bg-blue-700"
                          >
                            {downloadingVectorDb === db.id ? (
                              <Loader2 size={12} className="mr-1 animate-spin" />
                            ) : (
                              <Download size={12} className="mr-1" />
                            )}
                            Download
                          </Button>
                        </div>
                      )}
                    </DropdownMenuItem>
                  ))
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div className="flex items-center space-x-2">
            {/* Theme Toggle */}
            <Button variant="ghost" size="icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}
            </Button>
            {/* Settings Dropdown/Link (Placeholder) */}
            {/* <Button variant="ghost" size="icon">
              <Settings size={18}/>
            </Button> */}
            {/* User Info & Logout - Now uses the correctly typed user from context */}
            {user && <span className='text-sm'>{user.username}</span>}
            <Button variant="ghost" size="icon" onClick={logout} title="Logout">
              <LogOut size={18}/>
            </Button>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 flex overflow-y-auto">
          {/* Make children take full width/height */}
          <div className="flex-1 flex flex-col p-4 md:p-6">
             {children}
          </div>
        </main>
      </div>

      {/* Right Sidebar Removed */}
    </div>
  );
}
