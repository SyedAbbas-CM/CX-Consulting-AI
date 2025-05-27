"use client"

import React from 'react'
import { useEffect, useState } from "react"
import { useAuth } from "@/context/auth-context"
import ProtectedRoute from "@/components/protected-route"
import ChatInterface from "@/components/chat-interface"
import DocumentPreview from "@/components/document-preview"
import { useTheme } from "next-themes"
import { useChatStore } from "@/store/chatStore"
import { Button } from "@/components/ui/button"
import { PanelLeftClose, PanelRightClose, PanelLeftOpen, PanelRightOpen } from "lucide-react"

export default function DashboardPage() {
  const { theme } = useTheme()
  const { user } = useAuth()

  // Get project/chat state from the Zustand store
  const currentProjectId = useChatStore((state) => state.currentProjectId)
  // const currentChatId = useChatStore((state) => state.currentChatId)

  const [isDocPreviewCollapsed, setIsDocPreviewCollapsed] = useState(false);

  // Example: Logic to potentially load initial projects/chats if store is empty
  // Could be placed in a layout component or here
  useEffect(() => {
    if (!currentProjectId) {
      // TODO: Fetch user projects and set the first one as current?
      console.log("Dashboard: No current project selected in store.");
      // Example: useChatStore.getState().setCurrentProjectId('proj_fetched_123');
    }
  }, [currentProjectId]);

  return (
    <ProtectedRoute>
      <div className="flex flex-col h-screen bg-gray-900"> {/* Base background for dark theme */}
        {/* Header */}
        <header className="p-4 bg-gray-900 border-b border-gray-800 flex-shrink-0 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-semibold text-gray-100">CX Consulting AI Dashboard</h1>
          {/* TODO: Add Project Selection Dropdown Here */}
          {/* Display project ID from store */}
            <p className="text-xs text-gray-400">Current Project: {currentProjectId ? currentProjectId.substring(0,8)+'...' : 'None Selected'}</p>
          </div>
          {currentProjectId && (
            <Button
              variant="outline"
              size="icon"
              onClick={() => setIsDocPreviewCollapsed(!isDocPreviewCollapsed)}
              title={isDocPreviewCollapsed ? "Open Document Panel" : "Collapse Document Panel"}
              className="text-gray-400 border-gray-700 hover:bg-gray-700 hover:text-gray-100"
            >
              {isDocPreviewCollapsed ? <PanelLeftOpen className="h-5 w-5" /> : <PanelRightClose className="h-5 w-5" />}
            </Button>
          )}
        </header>

        {/* Main content area - Grid layout */}
        <main className={`flex-grow grid gap-0 overflow-hidden ${isDocPreviewCollapsed || !currentProjectId ? 'grid-cols-1' : 'grid-cols-[1fr_1fr] md:grid-cols-[minmax(22rem,1fr)_minmax(22rem,1fr)]'}`}>
          {/* Chat Area */}
          <section className="flex flex-col bg-gray-900 border-r border-gray-800 overflow-y-auto min-w-[22rem]">
             {/* Render ChatInterface if project is selected */}
             {currentProjectId ? (
                 // Pass only the expected 'theme' prop, ensuring correct type
                 <ChatInterface theme={theme === 'dark' ? 'dark' : 'light'} />
             ) : (
                 <div className="flex-1 flex items-center justify-center text-gray-500 italic p-6 text-center">
                     Select or create a project and chat to begin.
                 </div>
             )}
          </section>

          {/* Document Preview Area (Right Panel) */}
          {(!isDocPreviewCollapsed && currentProjectId) && (
            <section className="flex flex-col bg-gray-800 overflow-y-auto shadow-inner p-6 space-y-4 min-w-[22rem] text-gray-300">
              {/* DocumentPreview component will handle its own content and scroll */}
                  <DocumentPreview theme={theme === 'dark' ? 'dark' : 'light'} />
            </section>
              )}
        </main>

        {/* Footer - Chat Input is likely within ChatInterface now */}
        {/* Footer is removed as input handling is inside ChatInterface */}
        {/* <footer className="p-4 bg-background border-t flex-shrink-0"> */}
        {/*    ... potential footer content ... */}
        {/* </footer> */}
      </div>
    </ProtectedRoute>
  )
}
