import { Button } from "@/components/ui/button"
import { Download, Share2, Printer, Copy, Edit } from "lucide-react"

interface DocumentToolsProps {
  theme: "light" | "dark"
  visible: boolean
}

export default function DocumentTools({ theme, visible }: DocumentToolsProps) {
  if (!visible) return null

  return (
    <div
      className={`absolute top-16 right-4 flex flex-col gap-2 ${theme === "dark" ? "text-gray-300" : "text-gray-700"}`}
    >
      <Button variant="ghost" size="icon" className="rounded-full">
        <Download size={18} />
      </Button>
      <Button variant="ghost" size="icon" className="rounded-full">
        <Share2 size={18} />
      </Button>
      <Button variant="ghost" size="icon" className="rounded-full">
        <Printer size={18} />
      </Button>
      <Button variant="ghost" size="icon" className="rounded-full">
        <Copy size={18} />
      </Button>
      <Button variant="ghost" size="icon" className="rounded-full">
        <Edit size={18} />
      </Button>
    </div>
  )
}
