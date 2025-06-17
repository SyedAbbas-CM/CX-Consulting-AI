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
import { ChatCreateRequest } from "../types/chat";

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
  const [newProjectData, setNewProjectData] = useState<{name?: string; industry?: string; description?: string; goals?: string}>({});
  const { toast } = useToast();

  // New Chat (Journey) dialog state
  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatData, setNewChatData] = useState<{name?: string; journey_type?: string}>({});

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
      const payload = {
        name: newProjectData.name!.trim(),
        industry: newProjectData.industry?.trim() || undefined,
        description: newProjectData.description?.trim() || "",
        metadata: {
          goals: newProjectData.goals?.split(/,\s*/) || undefined,
        }
      } as unknown as ProjectCreateRequest;
      const project = await createProject(payload);
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

  // Open New Chat Dialog
  const openNewChatDialog = () => {
    if (!currentProjectId) {
      toast({ variant: "destructive", title: "No Project Selected", description: "Please select a project first." });
      return;
    }
    setShowNewChatDialog(true);
  };

  // Handle New Chat Submit
  const handleNewChatSubmit = async () => {
    if (!currentProjectId) return;
    if (!newChatData.name || !newChatData.name.trim()) {
      toast({ variant: "destructive", title: "Name Required", description: "Please enter a name for the journey." });
      return;
    }
    setIsCreatingChat(true);
    try {
      const payload = {
        name: newChatData.name.trim(),
        journey_type: newChatData.journey_type?.trim() || undefined,
      } as unknown as ChatCreateRequest;
      const chat = await createChat(currentProjectId, payload);
      addChat(chat);
      setCurrentChatId(chat.chat_id);
      setShowNewChatDialog(false);
      setNewChatData({});
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
          onClick={openNewChatDialog}
          disabled={!currentProjectId || isCreatingChat}
          className={`mb-2 w-full justify-center ${isCollapsed ? "p-0 h-9 w-9" : ""}`}
          title={isCollapsed ? "New Chat" : undefined}
        >
          {isCreatingChat ? <Loader2 className="animate-spin" /> : <PlusCircle />}
          {!isCollapsed && <span className="ml-2">New Journey</span>}
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
                  <div className="p-4 space-y-3">
                    <div>
                      <Label htmlFor="projectName">Name*</Label>
                      <Input
                        id="projectName"
                        value={newProjectData.name || ""}
                        onChange={(e) => setNewProjectData({ ...newProjectData, name: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="industry">Industry</Label>
                      <Input
                        id="industry"
                        placeholder="e.g. Retail, Banking"
                        value={newProjectData.industry || ""}
                        onChange={(e) => setNewProjectData({ ...newProjectData, industry: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="description">Description</Label>
                      <Input
                        id="description"
                        placeholder="Short description"
                        value={newProjectData.description || ""}
                        onChange={(e) => setNewProjectData({ ...newProjectData, description: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="goals">Goals (comma-separated)</Label>
                      <Input
                        id="goals"
                        placeholder="Improve NPS, reduce churn"
                        value={newProjectData.goals || ""}
                        onChange={(e) => setNewProjectData({ ...newProjectData, goals: e.target.value })}
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

        {/* New Chat Dialog */}
        <Dialog open={showNewChatDialog} onOpenChange={setShowNewChatDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Journey</DialogTitle>
              <DialogDescription>Fill details for the new customer journey.</DialogDescription>
            </DialogHeader>
            <div className="p-4 space-y-3">
              <div>
                <Label htmlFor="chatName">Name*</Label>
                <Input
                  id="chatName"
                  value={newChatData.name || ""}
                  onChange={e => setNewChatData({ ...newChatData, name: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="journeyType">Journey Type</Label>
                <Input
                  id="journeyType"
                  placeholder="e.g. roi_analysis, interview_prep"
                  value={newChatData.journey_type || ""}
                  onChange={e => setNewChatData({ ...newChatData, journey_type: e.target.value })}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="secondary" onClick={() => setShowNewChatDialog(false)}>Cancel</Button>
              <Button onClick={handleNewChatSubmit} disabled={isCreatingChat}>
                {isCreatingChat ? <Loader2 className="animate-spin mr-2" size={16}/> : null}
                Create Journey
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </aside>
  );
}
