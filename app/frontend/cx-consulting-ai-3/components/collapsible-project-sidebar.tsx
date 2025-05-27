"use client";

import React, { useState, useEffect } from "react";
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

export default function CollapsibleProjectSidebar({ theme }: CollapsibleProjectSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Project store
  const {
    projects,
    currentProjectId,
    isLoadingProjects,
    fetchProjects,
    addProject,
    removeProject,
    setCurrentProjectId,
  } = useProjectStore();

  // Chat store
  const {
    chats,
    currentChatId,
    isLoadingChats,
    fetchChats,
    addChat,
    removeChat,
    clearChats,
    setCurrentChatId,
    setProjectId: setChatStoreProjectId,
  } = useChatStore();

  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);
  const [newProjectData, setNewProjectData] = useState<Partial<ProjectCreateRequest>>({});
  const { toast } = useToast();

  // Load projects on mount
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // When project changes, update chat store and load its chats
  useEffect(() => {
    setChatStoreProjectId(currentProjectId);
    if (currentProjectId) {
      fetchChats(currentProjectId);
    } else {
      clearChats?.();
    }
  }, [currentProjectId, fetchChats, setChatStoreProjectId, clearChats]);

  // Create a new project
  const handleNewProjectSubmit = async () => {
    if (!newProjectData.name?.trim()) {
      toast({ variant: "destructive", title: "Missing Information", description: "Please enter a project name." });
      return;
    }
    setIsCreatingProject(true);
    try {
      const project = await createProject({ name: newProjectData.name.trim() } as ProjectCreateRequest);
      addProject(project);
      setCurrentProjectId(project.id);
      setShowNewProjectDialog(false);
      setNewProjectData({});
      toast({ title: "Project Created", description: `"${project.name}" created.` });
    } catch (err: any) {
      toast({ variant: "destructive", title: "Creation Failed", description: err?.errorData?.detail || err?.message });
    } finally {
      setIsCreatingProject(false);
    }
  };

  // Create a new chat under the current project
  const handleNewChat = async () => {
    if (!currentProjectId) {
      toast({ variant: "destructive", title: "No Project Selected", description: "Please select a project first." });
      return;
    }
    setIsCreatingChat(true);
    try {
      const chat = await createChat(currentProjectId);
      addChat(chat);
      setCurrentChatId(chat.chat_id);
    } catch (err: any) {
      toast({ variant: "destructive", title: "Chat Creation Failed", description: err?.errorData?.detail || err?.message });
    } finally {
      setIsCreatingChat(false);
    }
  };

  // Delete a chat
  const handleDeleteChat = async (chatId: string) => {
    if (!currentProjectId || deletingChatId) return;
    if (!window.confirm("Delete this chat?")) return;
    setDeletingChatId(chatId);
    try {
      await deleteChat(currentProjectId, chatId);
      removeChat(chatId);
    } catch (err: any) {
      toast({ variant: "destructive", title: "Deletion Failed", description: err?.errorData?.detail || err?.message });
    } finally {
      setDeletingChatId(null);
    }
  };

  // Delete a project and its chats
  const handleDeleteProject = async (projId: string, name: string) => {
    if (!window.confirm(`Delete project "${name}" and all its chats?`)) return;
    try {
      await deleteProject(projId);
      removeProject(projId);
      if (currentProjectId === projId) {
        setCurrentProjectId(null);
        clearChats?.();
      }
      toast({ title: "Project Deleted", description: `"${name}" removed.` });
    } catch (err: any) {
      toast({ variant: "destructive", title: "Deletion Failed", description: err?.errorData?.detail || err?.message });
    }
  };

  // Render the list of projects
  const renderProjects = () => {
    if (isLoadingProjects) return <p className="py-2 text-center text-xs text-gray-500">Loading projects…</p>;
    if (!projects.length) return <p className="py-2 text-center text-xs text-gray-500">(No projects)</p>;
    return projects.map((p) => (
      <AlertDialog key={p.id}>
        <div
          onClick={() => setCurrentProjectId(p.id)}
          className={`group flex items-center justify-between p-2 mb-1 rounded cursor-pointer text-sm
            ${currentProjectId === p.id ? (theme === "dark" ? "bg-blue-800 text-white" : "bg-blue-100 text-blue-900") : (theme === "dark" ? "hover:bg-gray-700 text-gray-300" : "hover:bg-gray-100 text-gray-700")}
          `}
        >
          <Folder className="mr-2 text-blue-500" />
          <span className="truncate flex-1">{p.name}</span>
          <AlertDialogTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="opacity-0 group-hover:opacity-100 text-red-500"
              onClick={(e) => { e.stopPropagation(); }}
            >
              <Trash2 />
            </Button>
          </AlertDialogTrigger>
        </div>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete "{p.name}" and all its chats.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => handleDeleteProject(p.id, p.name)}>
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    ));
  };

  // Render the list of chats for the active project
  const chatsForCurrent = chats.filter(c => c.project_id === currentProjectId);
  console.log("Sidebar: rendering", chatsForCurrent.length, "chats for project", currentProjectId);

  const renderChats = () => {
    if (!currentProjectId) return <p className="py-2 text-center text-xs text-gray-500">(Select a project first)</p>;
    if (isLoadingChats) return <p className="py-2 text-center text-xs text-gray-500">Loading chats…</p>;
    if (!chatsForCurrent.length) return <p className="py-2 text-center text-xs text-gray-500">(No chats)</p>;
    return chatsForCurrent.map((c) => (
      <div
        key={c.chat_id}
        onClick={() => setCurrentChatId(c.chat_id)}
        className={`group flex items-center p-2 mb-1 rounded cursor-pointer text-sm truncate
          ${currentChatId === c.chat_id ? (theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-200 text-gray-900") : (theme === "dark" ? "hover:bg-gray-800 text-gray-300" : "hover:bg-gray-100 text-gray-700")}
        `}
      >
        <MessageSquare className="mr-2" />
        <span className="truncate">{c.name}</span>
        <Button
          variant="ghost"
          size="icon"
          className="ml-auto opacity-0 group-hover:opacity-100 text-red-500"
          onClick={(e) => { e.stopPropagation(); handleDeleteChat(c.chat_id); }}
        >
          {deletingChatId === c.chat_id ? <Loader2 className="animate-spin" /> : <Trash2 />}
        </Button>
      </div>
    ));
  };

  return (
    <aside className={`flex flex-col h-full border-r transition-all ease-in-out duration-300
      ${theme === "dark" ? "bg-gray-900 border-gray-700" : "bg-gray-50 border-gray-300"}
      ${isCollapsed ? "w-16" : "w-64"}
    `}>
      <div className="flex items-center justify-between p-2">
        {!isCollapsed && <span className="font-semibold text-sm">Projects & Chats</span>}
        <Button variant="ghost" size="icon" onClick={() => setIsCollapsed(!isCollapsed)}>
          {isCollapsed ? <ChevronsRight /> : <ChevronsLeft />}
        </Button>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden p-2">
        <Button
          onClick={handleNewChat}
          disabled={!currentProjectId || isCreatingChat}
          className={`mb-2 w-full justify-center ${isCollapsed ? "p-0 h-9 w-9" : ""}`}
          title={isCollapsed ? "New Chat" : undefined}
        >
          {isCreatingChat ? <Loader2 className="animate-spin" /> : <PlusCircle />}
          {!isCollapsed && <span className="ml-2">New Chat</span>}
        </Button>

        {!isCollapsed && <hr className={`${theme === "dark" ? "border-gray-700" : "border-gray-300"} mb-2`} />}

        <ScrollArea className="flex-1 overflow-auto">
          {!isCollapsed && (
            <div className="mb-4">
              <Dialog open={showNewProjectDialog} onOpenChange={setShowNewProjectDialog}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-xs uppercase text-gray-500">Projects</h3>
                  <DialogTrigger asChild>
                    <Button size="icon">
                      {isCreatingProject ? <Loader2 className="animate-spin" /> : <PlusCircle />}
                    </Button>
                  </DialogTrigger>
                </div>
                {renderProjects()}

                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Create Project</DialogTitle>
                    <DialogDescription>Enter a name for the new project.</DialogDescription>
                  </DialogHeader>
                  <div className="p-4">
                    <div className="mb-2">
                      <Label htmlFor="projectName">Name*</Label>
                      <Input
                        id="projectName"
                        value={newProjectData.name || ""}
                        onChange={(e) => setNewProjectData({ name: e.target.value })}
                      />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button onClick={handleNewProjectSubmit} disabled={isCreatingProject}>
                      {isCreatingProject ? <Loader2 className="mr-2 animate-spin" /> : null}Create
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          )}

          {!isCollapsed && (
            <div>
              <h3 className="font-semibold text-xs uppercase text-gray-500 mb-2">Chats</h3>
              {renderChats()}
            </div>
          )}
        </ScrollArea>
      </div>
    </aside>
  );
}
