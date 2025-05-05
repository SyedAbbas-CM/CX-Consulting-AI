import { FileText } from "lucide-react"
import DocumentTools from "./document-tools"

interface DocumentPreviewProps {
  theme: "light" | "dark"
}

export default function DocumentPreview({ theme }: DocumentPreviewProps) {
  return (
    <div
      className={`flex flex-col border rounded-lg overflow-hidden ${
        theme === "dark" ? "border-gray-800 bg-[#0f1117]" : "border-gray-200 bg-gray-50"
      }`}
      style={{ height: "100%" }}
    >
      <div className="p-3 border-b">
        <h3 className="font-medium">Document Preview</h3>
        <p className={`text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
          Generated document will appear here
        </p>
      </div>
      <div className="relative">
        <DocumentTools theme={theme} visible={false} />
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div
          className={`w-24 h-24 flex items-center justify-center rounded-full mb-6 ${
            theme === "dark" ? "bg-gray-800" : "bg-gray-200"
          }`}
        >
          <FileText size={40} className={theme === "dark" ? "text-gray-400" : "text-gray-500"} />
        </div>
        <h4 className="text-xl font-medium mb-3">No document generated yet</h4>
        <p className={`text-center text-sm max-w-md ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
          Use the form below or ask the AI assistant to generate a document for you
        </p>
      </div>
    </div>
  )
}
