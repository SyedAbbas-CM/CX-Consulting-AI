"use client"

import React, { useState, useEffect } from 'react';
import { useTheme } from 'next-themes'; // Import useTheme

// Import Model Management related things
import { 
  getLlmConfig, 
  downloadModel,
  setActiveModel,
  listModels
} from '@/lib/apiClient';
import { LlmConfigResponse } from '@/types/config';
import { ModelInfo } from '@/types/model';
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

  // --- Effects from Sidebar for Model Management ---
  useEffect(() => {
    const fetchModelsAndConfig = async () => {
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
        } else {
          console.warn("DashboardLayout: Not fetching models/config, no auth token found.");
          // Handle state if not logged in? Maybe clear models/config?
          setModels([]);
          setLlmConfig(null);
          setActiveModelPath('Unknown');
        }
      } catch (error) {
        console.error("DashboardLayout: Error fetching models or config:", error);
        toast({ variant: "destructive", title: "Load Failed", description: "Could not load model or configuration data." });
        setModels([]);
        setLlmConfig(null); // Ensure config is null on error
      } finally {
        setLoadingModels(false);
        setIsLoadingConfig(false);
      }
    };
    fetchModelsAndConfig();
    // Add toast dependency
  }, [toast, user]); // Refetch if user changes (login/logout)

  // Effect to set mounted state
  useEffect(() => {
    setMounted(true);
  }, []);

  // --- Handlers from Sidebar for Model Management ---
  const handleDownload = async (modelId: string) => {
    setDownloadingModel(modelId);
    try {
      const result = await downloadModel(modelId);
      // alert(result.message);
      toast({ title: "Download Started", description: result.message });
      // Optionally refetch models after a delay or via websocket status?
    } catch (error: any) {
      console.error("Error downloading model:", error);
      const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
      // alert(`Failed to start model download: ${errorMessage}`);
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
                   ) : llmConfig ? (
                     <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
                   ) : (
                     <span className='text-red-500 font-bold'>!</span> // Indicate error state clearly
                   )}
                   <span className='font-mono truncate max-w-[200px]'>
                      {isLoadingConfig ? 'Loading...' : llmConfig?.model_id || activeModelPath || 'No Model Set'}
                   </span>
                   <ChevronsUpDown size={14} className="text-gray-500 flex-shrink-0" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-72">
                <DropdownMenuLabel>Model Management</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {/* Model List goes here */}
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
                          <div className="w-full flex justify-end">
                              <Button
                                  size="sm"
                                  variant="default"
                                  onClick={(e) => { e.stopPropagation(); handleDownload(model.id); }} // Stop propagation
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