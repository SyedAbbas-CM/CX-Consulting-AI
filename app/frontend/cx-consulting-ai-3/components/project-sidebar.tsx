"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Folder, PlusCircle, MessageSquare, Trash2, Loader2 } from "lucide-react"
import { useProjectStore } from '../store/projectStore'; // Import project store
import { useChatStore } from '../store/chatStore'; // Import chat store
import { createProject, createChat, deleteChat } from '../lib/apiClient'; // Import API functions
import { ProjectCreateRequest } from "../types/project"; // Import type
// Import UI components for the dialog
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription, 
  DialogFooter, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "./ui/use-toast"; // Import toast

interface ProjectSidebarProps {
  theme: "light" | "dark"
}

export default function ProjectSidebar({ theme }: ProjectSidebarProps) {
  // --- Zustand State --- 
  const { 
    projects, 
    currentProjectId, 
    isLoadingProjects, 
    fetchProjects, 
    addProject, 
    setCurrentProjectId 
  } = useProjectStore();

  const { 
    chats, 
    currentChatId, 
    isLoadingChats, 
    fetchChats, 
    addChat, 
    removeChat, 
    setCurrentChatId,
    setProjectId: setChatStoreProjectId // Rename to avoid conflict
  } = useChatStore();

  // --- Local Component State --- 
  const [isCreatingProject, setIsCreatingProject] = useState<boolean>(false);
  const [isCreatingChat, setIsCreatingChat] = useState<boolean>(false);
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState<boolean>(false);
  const [newProjectData, setNewProjectData] = useState<Partial<ProjectCreateRequest>>({});
  const { toast } = useToast();

  // --- Effects --- 
  // Fetch projects on component mount
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // Fetch chats when the current project changes
  useEffect(() => {
    setChatStoreProjectId(currentProjectId); // Update chat store's project ID
    if (currentProjectId) {
      fetchChats(currentProjectId);
    }
  }, [currentProjectId, fetchChats, setChatStoreProjectId]);

  // --- Handlers (Copied/Adapted from sidebar.tsx) --- 
  const handleNewProjectSubmit = async () => {
    // Validate required fields
    if (!newProjectData.name || !newProjectData.client_name || !newProjectData.industry || !newProjectData.description) {
      toast({ 
        variant: "destructive", 
        title: "Missing Information", 
        description: "Please fill out all required project fields."
      });
      return;
    }

    setIsCreatingProject(true);
    try {
      // Cast to ProjectCreateRequest after validation
      const projectData = newProjectData as ProjectCreateRequest;
      const newProject = await createProject(projectData);
      addProject(newProject);
      setCurrentProjectId(newProject.id); // Select the new project
      setShowNewProjectDialog(false); // Close dialog on success
      setNewProjectData({}); // Clear form
      toast({ title: "Project Created", description: `Project "${newProject.name}" created successfully.` });
    } catch (error: any) { 
      console.error("Failed to create project:", error);
      const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
      // alert(`Error creating project: ${errorMessage}`);
      toast({ variant: "destructive", title: "Creation Failed", description: `Error creating project: ${errorMessage}` });
    } finally {
      setIsCreatingProject(false);
    }
  };

  const handleNewChat = async () => {
    if (!currentProjectId) {
      alert("Please select a project first.");
      return;
    }
    setIsCreatingChat(true);
    try {
      const newChat = await createChat(currentProjectId);
      addChat(newChat);
      setCurrentChatId(newChat.id); // Select the new chat
    } catch (error: any) { 
      console.error("Failed to create new chat:", error);
      const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
      alert(`Error creating chat: ${errorMessage}`);
    } finally {
      setIsCreatingChat(false);
    }
  };

  const handleDeleteChat = async (chatIdToDelete: string, chatTitle: string) => {
    if (deletingChatId) return;
    // Ensure we have a project ID before attempting deletion
    if (!currentProjectId) {
      alert("Cannot delete chat: No project selected."); // Simple alert for this component
      return;
    }
    if (window.confirm(`Are you sure you want to delete the chat \"${chatTitle}\"?`)) {
      setDeletingChatId(chatIdToDelete);
      try {
        // Pass currentProjectId to the updated deleteChat function
        await deleteChat(currentProjectId, chatIdToDelete);
        removeChat(chatIdToDelete);
      } catch (error: any) { 
        console.error("Failed to delete chat:", error);
        const errorMessage = error?.errorData?.detail || error?.message || "An unknown error occurred.";
        alert(`Error deleting chat: ${errorMessage}`);
      } finally {
        setDeletingChatId(null);
      } 
    }
  };

  // --- JSX --- 
  return (
    <aside
      className={`w-72 border-r ${theme === "dark" ? "border-gray-800 bg-[#0f1117]" : "border-gray-200 bg-gray-50"} p-4 flex flex-col`} // Adjusted width/padding/colors slightly
    >
      {/* New Chat Button */}
      <Button
        className={`w-full mb-4 ${theme === "dark" ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600 text-white"}`}
        onClick={handleNewChat} // Use handler
        disabled={!currentProjectId || isCreatingChat} // Disable if no project or creating
      >
        {isCreatingChat ? (
            <Loader2 size={16} className="mr-2 animate-spin" />
        ) : (
            <PlusCircle size={16} className="mr-2" />
        )}
        New Chat
      </Button>

      {/* Wrap the trigger and content in the Dialog */}
      <Dialog open={showNewProjectDialog} onOpenChange={setShowNewProjectDialog}>
        {/* Projects Section (contains the trigger) */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-sm uppercase tracking-wider ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}">Projects</h3>
            {/* The trigger is now inside the Dialog */}
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7" disabled={isCreatingProject}>
                  {isCreatingProject ? <Loader2 size={16} className="animate-spin"/> : <PlusCircle size={18} />}
              </Button>
            </DialogTrigger>
          </div>
          {isLoadingProjects ? (
            <div className="text-center text-xs py-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}">Loading projects...</div>
          ) : (
            <div className="space-y-1">
              {projects.map((project) => (
                <div
                  key={project.id}
                  // Connect project selection to setCurrentProjectId
                  onClick={() => setCurrentProjectId(project.id)}
                  className={`p-2 rounded-md flex items-center cursor-pointer ${ 
                    currentProjectId === project.id ? 
                      (theme === "dark" ? "bg-gray-700" : "bg-gray-200") 
                      : (theme === "dark" ? "hover:bg-gray-800" : "hover:bg-gray-100")
                  }`}
                >
                  <Folder size={16} className="mr-2 text-blue-500 flex-shrink-0" />
                  <div className="truncate">
                    <p className="text-sm font-medium truncate" title={project.name}>{project.name}</p>
                    {/* Optionally fetch/display document count later */}
                    {/* <p className={`text-xs ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}> {project.documents} documents </p> */}
                  </div>
                </div>
              ))}
              {projects.length === 0 && <p className="text-xs text-center py-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}">(No projects yet)</p>}
            </div>
          )}
        </div>

        {/* Dialog Content (remains inside the Dialog) */}
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
            <DialogDescription>
              Enter the details for your new project. Click save when you're done.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="projectName" className="text-right">
                Name*
              </Label>
              <Input 
                id="projectName" 
                value={newProjectData.name || ''} 
                onChange={(e) => setNewProjectData({...newProjectData, name: e.target.value})} 
                className="col-span-3" 
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="clientName" className="text-right">
                Client Name*
              </Label>
              <Input 
                id="clientName" 
                value={newProjectData.client_name || ''} 
                onChange={(e) => setNewProjectData({...newProjectData, client_name: e.target.value})} 
                className="col-span-3" 
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="industry" className="text-right">
                Industry*
              </Label>
              <Input 
                id="industry" 
                value={newProjectData.industry || ''} 
                onChange={(e) => setNewProjectData({...newProjectData, industry: e.target.value})} 
                className="col-span-3" 
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="description" className="text-right">
                Description*
              </Label>
              <Textarea 
                id="description" 
                value={newProjectData.description || ''} 
                onChange={(e) => setNewProjectData({...newProjectData, description: e.target.value})} 
                className="col-span-3" 
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button 
              type="submit" 
              onClick={handleNewProjectSubmit} 
              disabled={isCreatingProject}
            >
              {isCreatingProject ? <Loader2 size={16} className="mr-2 animate-spin" /> : null}
              Save Project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog> { /* End Dialog Wrapper */ }

      {/* Chats Section */}
      <div className="flex-1 overflow-y-auto">
        <h3 className="font-semibold text-sm uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}">Chats</h3>
        {!currentProjectId ? (
            <p className="text-xs text-center py-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}">(Select a project)</p>
        ) : isLoadingChats ? (
            <div className="text-center text-xs py-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}">Loading chats...</div>
        ) : (
            <div className="space-y-1">
              {/* Filter chats for the currently selected project */} 
              {chats
                .filter(chat => chat.project_id === currentProjectId)
                .map((chat) => (
                <div
                  // Connect chat selection to setCurrentChatId
                  key={chat.id} 
                  className={`p-2 rounded-md flex items-center justify-between cursor-pointer group ${ 
                    currentChatId === chat.id ? 
                      (theme === "dark" ? "bg-gray-700" : "bg-gray-200") 
                      : (theme === "dark" ? "hover:bg-gray-800" : "hover:bg-gray-100")
                  }`}
                >
                  <div className="flex items-center truncate mr-1" onClick={() => setCurrentChatId(chat.id)}>
                    {/* <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center mr-2 flex-shrink-0">
                      <span className="text-white text-xs">AI</span>
                    </div> */} 
                    <MessageSquare size={16} className="mr-2 flex-shrink-0 text-gray-500"/>
                    <div className="truncate">
                      <p className="text-sm truncate" title={chat.title}>{chat.title}</p>
                      <p className={`text-xs ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
                          {new Date(chat.last_updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}, {new Date(chat.last_updated_at).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                  {/* Add Delete Button */} 
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className={`h-6 w-6 text-gray-500 hover:text-red-500 invisible group-hover:visible flex-shrink-0 ${deletingChatId === chat.id ? 'text-red-500' : ''}`}
                    onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id, chat.title); }} 
                    disabled={deletingChatId === chat.id}
                  >
                    {deletingChatId === chat.id ? <Loader2 size={14} className="animate-spin"/> : <Trash2 size={14} />}
                  </Button>
                </div>
              ))}
              {currentProjectId && chats.filter(chat => chat.project_id === currentProjectId).length === 0 && 
                  <p className="text-xs text-center py-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}">(No chats in this project)</p>}
            </div>
        )}
      </div>
    </aside>
  );
}
