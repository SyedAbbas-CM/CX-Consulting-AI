"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { MoonIcon, SunIcon, UserPlus, Mail, Lock, User, Eye, EyeOff } from "lucide-react"
import { useAuth } from "@/context/auth-context"

export default function RegisterPage() {
  const router = useRouter()
  const { register, isLoading } = useAuth()
  const [theme, setTheme] = useState<"light" | "dark">("dark")
  const [showPassword, setShowPassword] = useState(false)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [agreeTerms, setAgreeTerms] = useState(false)
  const [error, setError] = useState("")

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light")
  }

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!name || !email || !password) {
      setError("Please fill in all required fields")
      return
    }

    if (!agreeTerms) {
      setError("You must agree to the Terms of Service and Privacy Policy")
      return
    }

    try {
      const success = await register(name, email, password)

      if (success) {
        // Registration successful, redirect to login
        router.push("/login?registered=true")
      } else {
        setError("Registration failed. Please try again.")
      }
    } catch (err) {
      setError("Registration failed. Email may already be in use.")
    }
  }

  return (
    <div
      className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-[#0a0c14] text-white" : "bg-white text-black"}`}
    >
      {/* Header */}
      <header className={`border-b ${theme === "dark" ? "border-gray-800" : "border-gray-200"} py-3 px-6`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <span className="text-blue-500 font-bold text-xl mr-2">CX</span>
            <h1 className="text-xl font-bold">CX Consulting AI</h1>
          </div>
          <button onClick={toggleTheme} className="p-2 rounded-full">
            {theme === "dark" ? <SunIcon size={20} /> : <MoonIcon size={20} />}
          </button>
        </div>
      </header>

      {/* Register Form */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div
          className={`w-full max-w-md p-8 rounded-lg border ${
            theme === "dark" ? "bg-[#0f1117] border-gray-800" : "bg-white border-gray-200"
          }`}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-2">Create an Account</h2>
            <p className={theme === "dark" ? "text-gray-400" : "text-gray-600"}>
              Join CX Consulting AI to create professional deliverables
            </p>
          </div>

          {error && (
            <div className="bg-red-50 text-red-700 p-3 rounded-md mb-4">
              {error}
            </div>
          )}

          <form onSubmit={handleRegister}>
            <div className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="name" className="block text-sm font-medium">
                  Full Name
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                    <User size={18} className={theme === "dark" ? "text-gray-500" : "text-gray-400"} />
                  </div>
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className={`pl-10 ${theme === "dark" ? "bg-gray-800 border-gray-700" : ""}`}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="email" className="block text-sm font-medium">
                  Email
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                    <Mail size={18} className={theme === "dark" ? "text-gray-500" : "text-gray-400"} />
                  </div>
                  <Input
                    id="email"
                    type="email"
                    placeholder="name@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={`pl-10 ${theme === "dark" ? "bg-gray-800 border-gray-700" : ""}`}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="password" className="block text-sm font-medium">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                    <Lock size={18} className={theme === "dark" ? "text-gray-500" : "text-gray-400"} />
                  </div>
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className={`pl-10 ${theme === "dark" ? "bg-gray-800 border-gray-700" : ""}`}
                    required
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 flex items-center pr-3"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff size={18} className={theme === "dark" ? "text-gray-500" : "text-gray-400"} />
                    ) : (
                      <Eye size={18} className={theme === "dark" ? "text-gray-500" : "text-gray-400"} />
                    )}
                  </button>
                </div>
                <p className={`text-xs ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
                  Must be at least 8 characters and include a number and a special character
                </p>
              </div>

              <div className="flex items-start space-x-2">
                <Checkbox
                  id="terms"
                  checked={agreeTerms}
                  onCheckedChange={(checked) => setAgreeTerms(checked as boolean)}
                  className="mt-1"
                />
                <label htmlFor="terms" className="text-sm">
                  I agree to the{" "}
                  <a
                    href="#"
                    className={`font-medium ${
                      theme === "dark" ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-500"
                    }`}
                  >
                    Terms of Service
                  </a>{" "}
                  and{" "}
                  <a
                    href="#"
                    className={`font-medium ${
                      theme === "dark" ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-500"
                    }`}
                  >
                    Privacy Policy
                  </a>
                </label>
              </div>

              <Button
                type="submit"
                className={`w-full ${
                  theme === "dark" ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600"
                }`}
                disabled={isLoading || !agreeTerms}
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div className="h-4 w-4 animate-spin rounded-full border-b-2 border-white mr-2"></div>
                    Creating account...
                  </div>
                ) : (
                  <>
                    <UserPlus size={18} className="mr-2" />
                    Create Account
                  </>
                )}
              </Button>
            </div>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm">
              Already have an account?{" "}
              <Link
                href="/login"
                className={`font-medium ${
                  theme === "dark" ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-500"
                }`}
              >
                Sign in
              </Link>
            </p>
          </div>

          <div className={`mt-8 pt-6 border-t ${theme === "dark" ? "border-gray-800" : "border-gray-200"}`}>
            <p className={`text-center text-sm mb-4 ${theme === "dark" ? "text-gray-400" : "text-gray-600"}`}>
              Or sign up with
            </p>
            <div className="flex justify-center space-x-4">
              <Button
                variant="outline"
                className={`flex-1 ${
                  theme === "dark"
                    ? "bg-gray-800 border-gray-700 hover:bg-gray-700"
                    : "bg-white border-gray-200 hover:bg-gray-50"
                }`}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mr-2"
                >
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                </svg>
                GitHub
              </Button>
              <Button
                variant="outline"
                className={`flex-1 ${
                  theme === "dark"
                    ? "bg-gray-800 border-gray-700 hover:bg-gray-700"
                    : "bg-white border-gray-200 hover:bg-gray-50"
                }`}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mr-2"
                >
                  <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path>
                </svg>
                Google
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
