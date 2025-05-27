import { Button } from "@/components/ui/button"
import { Download, Share2, Printer, Copy, Edit } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface DocumentToolsProps {
  theme: "light" | "dark"
  visible: boolean
  onDownload?: () => void
  onShare?: () => void
  onPrint?: () => void
  onCopy?: () => void
  onEdit?: () => void
}

export default function DocumentTools({
  visible,
  onDownload,
  onShare,
  onPrint,
  onCopy,
  onEdit,
}: DocumentToolsProps) {
  if (!visible) return null

  const iconSize = 16
  const buttonClassName = "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"

  return (
    <TooltipProvider>
      <div className="flex items-center gap-1.5">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={onDownload} className={buttonClassName}>
              <Download size={iconSize} className="mr-1.5" /> Download
      </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Download Document</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={onShare} className={buttonClassName}>
              <Share2 size={iconSize} className="mr-1.5" /> Share
      </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Share Document</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={onPrint} className={buttonClassName}>
              <Printer size={iconSize} className="mr-1.5" /> Print
      </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Print Document</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={onCopy} className={buttonClassName}>
              <Copy size={iconSize} className="mr-1.5" /> Copy
      </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Copy Content</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={onEdit} className={buttonClassName}>
              <Edit size={iconSize} className="mr-1.5" /> Edit
      </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Edit Document</p>
          </TooltipContent>
        </Tooltip>
    </div>
    </TooltipProvider>
  )
}
