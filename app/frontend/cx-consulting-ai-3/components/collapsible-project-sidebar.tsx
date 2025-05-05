// CollapsibleProjectSidebar – rewritten to eliminate TypeScript & JSX errors
// Assumptions:
// 1. useProjectStore exposes removeProject
// 2. useChatStore exposes clearChats (to empty the list for a project) –
//    if that action doesn't exist, simply omit the call.
// 3. apiClient exports deleteProject (mirroring createProject / createChat / deleteChat)

"use client";

import { useState, useEffect } from "react";
import {
  Folder,
  PlusCircle,
  MessageSquare,
  Trash2,
  Loader2,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";

import { useProjectStore } from "../store/projectStore";
import { useChatStore } from "../store/chatStore";
import {
  createProject,
  createChat,
  deleteChat,
  deleteProject,
} from "../lib/apiClient";
import { ProjectCreateRequest } from "../types/project";

interface CollapsibleProjectSidebarProps {
  theme: "light" | "dark";
}

export default function CollapsibleProjectSidebar({
  theme,
}: CollapsibleProjectSidebarProps) {
  // layout – collapsed/expanded
  const [isCollapsed, setIsCollapsed] = useState(false);

  /** ------------------------------------------------------------------
   *  Zustand stores
   * ----------------------------------------------------------------*/
  const {
    projects,
    currentProjectId,
    isLoadingProjects,
    fetchProjects,
    addProject,
    removeProject, // assumed to exist
    setCurrentProjectId,
  } = useProjectStore();

  const {
    chats,
    currentChatId,
    isLoadingChats,
    fetchChats,
    addChat,
    removeChat,
    clearChats, // optional; fall‑back handled below
    setCurrentChatId,
    setProjectId: setChatStoreProjectId,
  } = useChatStore();

  /** ------------------------------------------------------------------
   *  Local state (dialog & loading flags)
   * ----------------------------------------------------------------*/
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);
  const [newProjectData, setNewProjectData] = useState<
    Partial<ProjectCreateRequest>
  >({});
  const { toast } = useToast();

  /** ------------------------------------------------------------------
   *  Effects – initialise projects & chats
   * ----------------------------------------------------------------*/
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  useEffect(() => {
    setChatStoreProjectId(currentProjectId);
    if (currentProjectId) {
      fetchChats(currentProjectId);
    }
  }, [currentProjectId, fetchChats, setChatStoreProjectId]);

  /** ------------------------------------------------------------------
   *  Handlers – create project / chat / delete chat / delete project
   * ----------------------------------------------------------------*/
  const handleNewProjectSubmit = async () => {
    if (!newProjectData.name?.trim()) {
      toast({
        variant: "destructive",
        title: "Missing Information",
        description: "Please enter a project name.",
      });
      return;
    }

    setIsCreatingProject(true);
    try {
      const newProject = await createProject({
        name: newProjectData.name.trim(),
      } as ProjectCreateRequest);

      addProject(newProject);
      setCurrentProjectId(newProject.id);
      setShowNewProjectDialog(false);
      setNewProjectData({});
      toast({
        title: "Project Created",
        description: `Project "${newProject.name}" created successfully.`,
      });
    } catch (err: any) {
      const msg = err?.errorData?.detail || err?.message || "Unknown error.";
      toast({
        variant: "destructive",
        title: "Creation Failed",
        description: msg,
      });
    } finally {
      setIsCreatingProject(false);
    }
  };

  const handleNewChat = async () => {
    console.log("handleNewChat triggered. Current Project ID:", currentProjectId);
    if (!currentProjectId) {
      console.log("No current project ID, showing toast.");
      toast({
        variant: "destructive",
        title: "No Project Selected",
        description: "Select a project first.",
      });
      return;
    }
    console.log("Setting isCreatingChat to true");
    setIsCreatingChat(true);
    try {
      console.log(`Calling createChat API for project: ${currentProjectId}`);
      const newChat = await createChat(currentProjectId);
      console.log("createChat API response:", newChat);
      if (!newChat || !newChat.id) { // Check if response is valid
          console.error("Invalid response received from createChat API", newChat);
          throw new Error("Received invalid data when creating chat.");
      }
      console.log(`Adding chat to store: ${JSON.stringify(newChat)}`);
      addChat(newChat);
      console.log(`Setting current chat ID in store: ${newChat.id}`);
      setCurrentChatId(newChat.id);
      console.log("handleNewChat completed successfully.");
    } catch (err: any) {
      console.error("Error in handleNewChat:", err);
      const msg = err?.errorData?.detail || err?.message || "Unknown error.";
      toast({
        variant: "destructive",
        title: "Chat Creation Failed",
        description: msg,
      });
    } finally {
      console.log("Setting isCreatingChat to false");
      setIsCreatingChat(false);
    }
  };

  const handleDeleteChat = async (chatId: string, chatTitle: string) => {
    if (deletingChatId) return;
    if (!currentProjectId) {
      toast({
        variant: "destructive",
        title: "Cannot Delete Chat",
        description: "No project is currently selected.",
      });
      return;
    }
    if (!window.confirm(`Delete chat "${chatTitle}"?`)) return;

    setDeletingChatId(chatId);
    try {
      await deleteChat(currentProjectId, chatId);
      removeChat(chatId);
    } catch (err: any) {
      const msg = err?.errorData?.detail || err?.message || "Unknown error.";
      toast({
        variant: "destructive",
        title: "Deletion Failed",
        description: msg,
      });
    } finally {
      setDeletingChatId(null);
    }
  };

  const handleDeleteProject = async (projectId: string, projectName: string) => {
    try {
      await deleteProject(projectId);
      removeProject(projectId);
      toast({
        title: "Project Deleted",
        description: `Project "${projectName}" removed.`,
      });

      let shouldRefetchProjects = true;

      if (currentProjectId === projectId) {
        setCurrentProjectId(null);
        // clear local chats since the project is gone
        clearChats?.();
        shouldRefetchProjects = false; // fetchProjects will be triggered by setCurrentProjectId(null) -> useEffect
      }

      // Explicitly refetch projects if the deleted one wasn't selected
      // If it WAS selected, the useEffect watching currentProjectId handles it.
      if (shouldRefetchProjects) {
           await fetchProjects();
      }

    } catch (err: any) {
      const msg = err?.errorData?.detail || err?.message || "Unknown error.";
      toast({
        variant: "destructive",
        title: "Deletion Failed",
        description: msg,
      });
    }
  };

  /** ------------------------------------------------------------------
   *  Render helpers
   * ----------------------------------------------------------------*/
  const renderProjects = () => {
    if (isLoadingProjects) {
      return (
        <p
          className={`text-center text-xs py-4 ${
            theme === "dark" ? "text-gray-500" : "text-gray-400"
          }`}
        >
          Loading projects…
        </p>
      );
    }

    if (projects.length === 0) {
      return (
        <p
          className={`text-xs text-center py-2 ${
            theme === "dark" ? "text-gray-500" : "text-gray-400"
          }`}
        >
          (No projects)
        </p>
      );
    }

    // Filter out any projects without a valid ID before mapping
    const validProjects = projects.filter(p => p && p.id);

    return validProjects.map((project) => (
      <AlertDialog key={String(project.id)}>
        <div
          onClick={() => setCurrentProjectId(project.id)}
          className={`group flex items-center justify-between rounded-md px-2 py-1.5 text-sm cursor-pointer hover:bg-opacity-80 ${
            currentProjectId === project.id
              ? theme === "dark"
                ? "bg-blue-900/50 text-white"
                : "bg-blue-100 text-blue-800"
              : theme === "dark"
              ? "hover:bg-gray-700"
              : "hover:bg-gray-100"
          }`}
        >
          <Folder size={16} className="mr-2 text-blue-500 flex-shrink-0" />
          <p className="truncate flex-1" title={project.name}>
            {project.name}
          </p>
          <AlertDialogTrigger asChild onClick={(e) => e.stopPropagation()}>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 text-red-500 hover:bg-red-500/10"
              title={`Delete project "${project.name}"`}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </AlertDialogTrigger>
        </div>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the project "{project.name}"? This
              will permanently delete the project and all associated chats.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => handleDeleteProject(project.id, project.name)}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete Project
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    ));
  };

  const chatsForCurrent = chats.filter(
    (c) => c.project_id === currentProjectId,
  );

  // Filter out any chats without a valid ID before mapping
  const validChatsForCurrent = chatsForCurrent.filter(c => c && c.id);

  const renderChats = () => {
    if (!currentProjectId) {
      return (
        <p
          className={`text-xs text-center py-2 ${
            theme === "dark" ? "text-gray-500" : "text-gray-400"
          }`}
        >
          (Select project)
        </p>
      );
    }
    if (isLoadingChats) {
      return (
        <p
          className={`text-center text-xs py-4 ${
            theme === "dark" ? "text-gray-500" : "text-gray-400"
          }`}
        >
          Loading chats…
        </p>
      );
    }
    if (chatsForCurrent.length === 0) {
      return (
        <p
          className={`text-xs text-center py-2 ${
            theme === "dark" ? "text-gray-500" : "text-gray-400"
          }`}
        >
          (No chats)
        </p>
      );
    }

    return validChatsForCurrent.map((chat) => (
      <div
        key={String(chat.id)}
        className={`p-2 rounded-md flex items-center justify-between cursor-pointer group ${
          currentChatId === chat.id
            ? theme === "dark"
              ? "bg-gray-700"
              : "bg-gray-200"
            : theme === "dark"
            ? "hover:bg-gray-800"
            : "hover:bg-gray-100"
        }`}
      >
        <div
          className="flex items-center truncate mr-1 flex-1"
          onClick={() => setCurrentChatId(chat.id)}
        >
          <MessageSquare
            size={16}
            className="mr-2 flex-shrink-0 text-gray-500"
          />
          <div className="truncate">
            <p className="truncate" title={chat?.title ?? ''}>
              {chat?.title || `Chat ${chat?.id ? chat.id.substring(0, 8) : '...'}`}
            </p>
            <p
              className={`text-xs ${
                theme === "dark" ? "text-gray-400" : "text-gray-500"
              }`}
            >
              {new Date(chat.last_updated_at).toLocaleString(undefined, {
                month: "short",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
              })}
            </p>
          </div>
        </div>

        <Button
          variant="ghost"
          size="icon"
          className={`h-6 w-6 text-gray-500 hover:text-red-500 invisible group-hover:visible flex-shrink-0 ${
            deletingChatId === chat.id ? "text-red-500" : ""
          }`}
          onClick={(e) => {
            e.stopPropagation();
            handleDeleteChat(
              chat.id,
              chat?.title || `Chat ${chat?.id ? chat.id.substring(0, 8) : '...'}`
            );
          }}
          disabled={deletingChatId === chat.id}
          title="Delete Chat"
        >
          {deletingChatId === chat.id ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Trash2 size={14} />
          )}
        </Button>
      </div>
    ));
  };

  /** ------------------------------------------------------------------
   *  JSX – main
   * ----------------------------------------------------------------*/
  return (
    <aside
      className={`relative border-r transition-all duration-300 ease-in-out ${
        theme === "dark" ? "border-gray-800 bg-[#0f1117]" : "border-gray-200 bg-gray-50"
      } ${isCollapsed ? "w-16" : "w-72"} flex flex-col`}
    >
      {/* collapse toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 -right-4 z-10 h-7 w-7 rounded-full border bg-background hover:bg-muted"
        onClick={() => setIsCollapsed(!isCollapsed)}
        title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
      >
        {isCollapsed ? <ChevronsRight size={16} /> : <ChevronsLeft size={16} />}
      </Button>

      {/* content */}
      <div
        className={`p-4 overflow-hidden flex-grow flex flex-col ${
          isCollapsed ? "items-center" : ""
        }`}
      >
        {/* New chat */}
        <Button
          className={`w-full mb-4 ${isCollapsed ? "h-10 w-10 p-0" : ""}`}
          onClick={handleNewChat}
          disabled={!currentProjectId || isCreatingChat}
          title={isCollapsed ? "New Chat" : ""}
        >
          {isCreatingChat ? (
            <Loader2
              size={16}
              className={`animate-spin ${!isCollapsed ? "mr-2" : ""}`}
            />
          ) : (
            <PlusCircle
              size={16}
              className={`${!isCollapsed ? "mr-2" : ""}`}
            />
          )}
          {!isCollapsed && "New Chat"}
        </Button>

        {/* separator */}
        {!isCollapsed && (
          <hr
            className={`my-2 ${
              theme === "dark" ? "border-gray-700" : "border-gray-300"
            }`}
          />
        )}

        {/* scrollable body */}
        <ScrollArea className="flex-grow w-full">
          {/* projects */}
          {!isCollapsed && (
            <div className="mb-6">
              <Dialog
                open={showNewProjectDialog}
                onOpenChange={setShowNewProjectDialog}
              >
                {/* header row (title + add btn) */}
                <div className="flex items-center justify-between mb-2">
                  <h3
                    className={`font-semibold text-sm uppercase tracking-wider ${
                      theme === "dark" ? "text-gray-400" : "text-gray-500"
                    }`}
                  >
                    Projects
                  </h3>
                  <DialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      disabled={isCreatingProject}
                      title="New Project"
                    >
                      {isCreatingProject ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <PlusCircle size={18} />
                      )}
                    </Button>
                  </DialogTrigger>
                </div>
                {renderProjects()}

                {/* dialog content – create project */}
                <DialogContent className="sm:max-w-[425px]">
                  <DialogHeader>
                    <DialogTitle>Create New Project</DialogTitle>
                    <DialogDescription>
                      Enter a project name.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="projectName" className="text-right">
                        Name*
                      </Label>
                      <Input
                        id="projectName"
                        value={newProjectData.name || ""}
                        onChange={(e) =>
                          setNewProjectData({ name: e.target.value })
                        }
                        className="col-span-3"
                        required
                      />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      onClick={handleNewProjectSubmit}
                      disabled={isCreatingProject}
                    >
                      {isCreatingProject && (
                        <Loader2 size={16} className="mr-2 animate-spin" />
                      )}
                      Create Project
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          )}

          {/* chats */}
          {!isCollapsed && (
            <div className="flex-1">
              <h3
                className={`font-semibold text-sm uppercase tracking-wider mb-2 ${
                  theme === "dark" ? "text-gray-400" : "text-gray-500"
                }`}
              >
                Chats
              </h3>
              {renderChats()}
            </div>
          )}
        </ScrollArea>

        {/* collapsed toolbar */}
        {isCollapsed && (
          <div className="mt-auto flex flex-col items-center space-y-4 pt-4">
            <Dialog
              open={showNewProjectDialog}
              onOpenChange={setShowNewProjectDialog}
            >
              <DialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  disabled={isCreatingProject}
                  title="New Project"
                >
                  <PlusCircle size={18} />
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>Create New Project</DialogTitle>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="projectNameCollapsed" className="text-right">
                      Name*
                    </Label>
                    <Input
                      id="projectNameCollapsed"
                      value={newProjectData.name || ""}
                      onChange={(e) =>
                        setNewProjectData({ name: e.target.value })
                      }
                      className="col-span-3"
                      required
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    onClick={handleNewProjectSubmit}
                    disabled={isCreatingProject}
                  >
                    {isCreatingProject && (
                      <Loader2 size={16} className="mr-2 animate-spin" />
                    )}
                    Save
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        )}
      </div>
    </aside>
  );
}
