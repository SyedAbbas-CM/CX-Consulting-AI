'use client';

import React, { useRef, useState } from 'react';
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Paperclip, FileText, UploadCloud } from 'lucide-react'; // Icons
import { useToast } from "@/components/ui/use-toast";
// Import the new API client function for uploading documents
import { uploadDocument } from '@/lib/apiClient'; // Adjust path if needed
// Import DocumentUploadResponse type if you want to type the result specifically
// import { DocumentUploadResponse } from '@/src/types/api';

interface ActionMenuProps {
  currentProjectId: string | null;
  onOpenDeliverableGenerator?: (deliverableType: string) => void; // Added callback
}

// Update to actual deliverable types and friendly names
const ACTUAL_DELIVERABLES = [
  { id: "cx_strategy", name: "CX Strategy" },
  { id: "roi_analysis", name: "ROI Analysis" },
  { id: "journey_map", name: "Customer Journey Map" },
  { id: "proposal", name: "Proposal Document" }, // Example, adjust if needed
];

export function ActionMenu({ currentProjectId, onOpenDeliverableGenerator }: ActionMenuProps) {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);

  // authToken will be handled by fetchWithAuth in apiClient.ts

  const handleDeliverableSelect = (deliverableId: string) => {
    const selectedDeliverable = ACTUAL_DELIVERABLES.find(d => d.id === deliverableId);
    if (onOpenDeliverableGenerator && selectedDeliverable) {
      onOpenDeliverableGenerator(selectedDeliverable.id);
    } else {
      // Fallback toast if the callback isn't provided, though it should be
      toast({
        title: "Action Menu Info",
        description: `Selected: ${selectedDeliverable?.name}. Modal trigger not connected. Pass type: '${deliverableId}'.`,
      });
    }
  };

  const handleUploadClick = () => {
    if (!currentProjectId) {
      toast({
        title: "Project Not Selected",
        description: "Please select or create a project before uploading files.",
        variant: "destructive",
      });
      return;
    }
    fileInputRef.current?.click(); // Trigger hidden file input
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!currentProjectId) {
        toast({ title: "Error", description: "No project ID selected for upload.", variant: "destructive" });
        return;
    }

    setIsUploading(true);

    try {
      // Call the new uploadDocument function from apiClient
      // The third parameter `isGlobal` is false by default for project-specific uploads here.
      const result = await uploadDocument(file, currentProjectId, false);
      // Assuming result is DocumentUploadResponse: { filename, document_id, project_id, is_global, chunks_created }

      toast({
        title: "File Uploaded Successfully",
        description: `${result.filename} has been uploaded to project ${result.project_id}. Document ID: ${result.document_id}`,
      });
    } catch (error: any) {
      console.error("Error uploading file:", error);
      const errorDetail = error.errorData?.detail || error.message || "An unexpected error occurred during upload.";
      toast({
        title: "Upload Failed",
        description: errorDetail,
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div className="flex items-center">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="icon" className="mr-2" title="Attach file or generate document">
            <Paperclip className="h-5 w-5" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuLabel>Actions</DropdownMenuLabel>
          <DropdownMenuSeparator />
           <DropdownMenuItem onClick={handleUploadClick} disabled={isUploading || !currentProjectId} className="flex items-center">
            <UploadCloud className="mr-2 h-4 w-4" />
            {isUploading ? "Uploading..." : "Upload Document to Project"}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuLabel>Generate Document</DropdownMenuLabel>
          {ACTUAL_DELIVERABLES.map((del) => (
            <DropdownMenuItem key={del.id} onClick={() => handleDeliverableSelect(del.id)} className="flex items-center">
              <FileText className="mr-2 h-4 w-4" />
              {del.name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".pdf,.txt,.doc,.docx,.csv,.xlsx,.md" // Updated accept attribute
        style={{ display: 'none' }}
      />
      {/* <Input className="flex-grow" placeholder="Type your message or command..." /> */}
    </div>
  );
}
