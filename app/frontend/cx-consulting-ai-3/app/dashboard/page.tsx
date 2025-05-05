"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/context/auth-context"
import ProtectedRoute from "@/components/protected-route"
import ChatInterface from "@/components/chat-interface"
import DocumentPreview from "@/components/document-preview"
import { useTheme } from "next-themes"

export default function DashboardPage() {
  const { theme } = useTheme()
  const { user } = useAuth()

  return (
    <ProtectedRoute>
      <div className="flex flex-1 flex-col md:flex-row gap-6 h-full">
        <div className="flex-1 h-full">
          <ChatInterface theme={theme === "light" ? "light" : "dark"} />
        </div>

        <div className="w-1/3 h-full flex-shrink-0">
          <DocumentPreview theme={theme === "light" ? "light" : "dark"} />
        </div>
      </div>
    </ProtectedRoute>
  )
} 