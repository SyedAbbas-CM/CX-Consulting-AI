"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/context/auth-context"
import { DeliverableGenerator } from "@/components/DeliverableGenerator"

export default function Home() {
  const router = useRouter()
  const { isAuthenticated, isLoading } = useAuth()

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        router.push("/dashboard")
      } else {
        router.push("/login")
      }
    }
  }, [isLoading, isAuthenticated, router])

  // This is just a loading page while authentication is checked
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-blue-600 mb-4"></div>
        <h2 className="text-xl font-medium text-gray-700">Loading CX Consulting AI...</h2>
      </div>
    </div>
  )
}
