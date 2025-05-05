"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import * as api from '@/lib/api'
import { User as AuthUser } from '@/types/auth'

interface AuthContextType {
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (name: string, email: string, password: string) => Promise<boolean>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false)
  const router = useRouter()

  // Check if user is already logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Check if token exists in localStorage
        const token = localStorage.getItem("authToken")
        if (!token) {
          setIsLoading(false)
          return
        }

        // Verify token with backend
        try {
          const userData = await api.getCurrentUser()
          setUser({
            id: userData.id,
            username: userData.username,
            email: userData.email,
            full_name: userData.full_name,
            disabled: userData.disabled,
          })
          setIsAuthenticated(true)
        } catch (error) {
          // If token is invalid, clear storage
          localStorage.removeItem("authToken")
          console.error("Token verification failed:", error)
        }
      } catch (error) {
        console.error("Auth verification error:", error)
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string): Promise<boolean> => {
    setIsLoading(true)
    try {
      const authResponse = await api.login(email, password)
      
      if (authResponse && authResponse.access_token) {
        localStorage.setItem("authToken", authResponse.access_token)
        
        // Get user data
        const userData = await api.getCurrentUser()
        setUser({
          id: userData.id,
          username: userData.username,
          email: userData.email,
          full_name: userData.full_name,
          disabled: userData.disabled,
        })
        
        setIsAuthenticated(true)
        setIsLoading(false)
        return true
      } else {
        setIsLoading(false)
        return false
      }
    } catch (error) {
      console.error("Login error:", error)
      setIsLoading(false)
      return false
    }
  }

  const register = async (name: string, email: string, password: string): Promise<boolean> => {
    setIsLoading(true)
    try {
      const userData = await api.register({
        username: email,
        email,
        password,
        full_name: name,
        company: ''
      })

      if (userData) {
        setIsLoading(false)
        // After registration, redirect to login
        return true
      } else {
        setIsLoading(false)
        return false
      }
    } catch (error) {
      console.error("Registration error:", error)
      setIsLoading(false)
      return false
    }
  }

  const logout = () => {
    localStorage.removeItem("authToken")
    setUser(null)
    setIsAuthenticated(false)
    router.push("/login")
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
} 