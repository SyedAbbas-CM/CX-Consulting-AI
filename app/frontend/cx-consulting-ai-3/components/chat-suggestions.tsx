import { Button } from "@/components/ui/button"

interface ChatSuggestionsProps {
  theme: "light" | "dark"
}

export default function ChatSuggestions({ theme }: ChatSuggestionsProps) {
  const suggestions = [
    "Create a customer journey map for an e-commerce website",
    "Generate a CX strategy for a SaaS company",
    "Analyze customer feedback and provide insights",
    "Draft a UX research plan for a mobile app",
  ]

  return (
    <div className="flex flex-wrap gap-2 mt-2 mb-4">
      {suggestions.map((suggestion, index) => (
        <Button
          key={index}
          variant="outline"
          size="sm"
          className={`text-xs ${theme === "dark" ? "bg-gray-800 border-gray-700 hover:bg-gray-700" : "bg-gray-100 border-gray-200 hover:bg-gray-200"}`}
        >
          {suggestion}
        </Button>
      ))}
    </div>
  )
}
