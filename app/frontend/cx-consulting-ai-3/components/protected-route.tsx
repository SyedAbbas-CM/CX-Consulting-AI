"use client"

import { useEffect } from "react"
import { useRouter, usePathname } from "next/navigation"
import { useAuth } from "@/context/auth-context"

interface ProtectedRouteProps {
  children: React.ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const router = useRouter()
  const pathname = usePathname()
  const { user, isLoading } = useAuth()

  useEffect(() => {
    // Skip auth check if already on login or register page
    const isAuthPage = pathname === "/login" || pathname === "/register"

    if (!isLoading && !user && !isAuthPage) {
      router.push("/login")
    }
  }, [user, isLoading, router, pathname])

  // Show loading state or render children
  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return <>{children}</>
}
