import { FileText, ListFilter, Loader2, Save } from "lucide-react"
import DocumentTools from "./document-tools"
import { useDocumentStore } from "../store/documentStore";
import { useEffect, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area"; // For document list
import { Button } from "@/components/ui/button"; // For document list items
import { Textarea } from "@/components/ui/textarea"; // Added Textarea
import { toast } from "@/components/ui/use-toast"; // For notifications

interface DocumentPreviewProps {
  theme: "light" | "dark"
}

export default function DocumentPreview({ theme }: DocumentPreviewProps) {
  const {
    documents,
    currentDocument,
    currentDocumentContent,
    isLoadingDocuments,
    isLoadingContent,
    error,
    currentProjectId,
    fetchDocumentsForProject,
    setCurrentDocumentById,
    updateCurrentDocumentContent,
    refineCurrentDocument
  } = useDocumentStore();

  const [editableContent, setEditableContent] = useState<string>("");
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [refinePrompt, setRefinePrompt] = useState<string>("");
  const [isRefining, setIsRefining] = useState<boolean>(false);

  useEffect(() => {
    // Only fetch documents when project ID changes, not when documents change
    if (currentProjectId && !isLoadingDocuments) {
      // Only fetch if we don't have documents for this project yet
      console.log("Document Preview: Project ID changed to", currentProjectId, ", fetching documents.");
      fetchDocumentsForProject(currentProjectId);
    } else if (!currentProjectId) {
      console.log("Document Preview: No project selected.");
    }
  }, [currentProjectId, fetchDocumentsForProject]); // Only depend on project ID and fetch function

  useEffect(() => {
    // When the store's currentDocumentContent changes (e.g., after fetch or save),
    // update the local editableContent and reset editing state.
    if (currentDocumentContent !== null) {
      setEditableContent(currentDocumentContent);
      setIsEditing(false); // Reset editing mode when new content is loaded/saved
    } else {
      setEditableContent(""); // Clear if no content
    }
  }, [currentDocumentContent]);

  const handleSelectDocument = (docId: string) => {
    if (isLoadingContent) return; // Don't change selection if content is already loading
    if (isEditing) {
        if (!window.confirm("You have unsaved changes. Are you sure you want to switch documents?")) {
            return;
        }
    }
    console.log("Document Preview: Selecting document", docId);
    setCurrentDocumentById(docId);
  };

  const handleContentChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditableContent(event.target.value);
    if (!isEditing) {
        setIsEditing(true);
    }
  };

  const handleSaveChanges = async () => {
    if (!currentDocument || !isEditing) return;
    if (isSaving) {
        toast({ title: "Saving...", description: "Please wait."});
        return;
    }
    setIsSaving(true);
    try {
      await updateCurrentDocumentContent(currentDocument.id, editableContent);
      toast({ title: "Success", description: "Document saved!" });
      setIsEditing(false);
    } catch (err: any) {
      toast({ variant: "destructive", title: "Save Failed", description: err.message || "Could not save document." });
    }
    setIsSaving(false);
  };

  const handleRefineDocument = async () => {
    if (!currentDocument || !refinePrompt) {
      toast({
        variant: "destructive",
        title: "Refinement Error",
        description: "Please select a document and enter a refinement prompt.",
      });
      return;
    }
    setIsRefining(true);
    try {
      await refineCurrentDocument(refinePrompt);
      toast({
        title: "Refinement Successful",
        description: "Document has been refined by AI.",
      });
      setRefinePrompt(""); // Clear prompt after successful refinement
    } catch (e: any) {
      toast({
        variant: "destructive",
        title: "Refinement Failed",
        description: e.message || "An unexpected error occurred during refinement.",
      });
    }
    setIsRefining(false);
  };

  // Displaying the list of documents and the content of the selected one.
  return (
    <div
      className={`flex flex-col border rounded-lg overflow-hidden ${
        theme === "dark" ? "border-gray-800 bg-[#0f1117]" : "border-gray-200 bg-gray-50"
      }`}
      style={{ height: "100%" }}
    >
      <div className="p-3 border-b flex justify-between items-center">
        <div>
          <h3 className="font-medium">Document Review</h3>
          <p className={`text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
            {currentDocument ? currentDocument.title : "Select a document"}
          </p>
        </div>
        <DocumentTools theme={theme} visible={!!currentDocument} />
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Document List Sidebar (within DocumentPreview) */}
        <ScrollArea className="w-1/3 border-r dark:border-gray-700 p-2">
          <h4 className="text-xs uppercase font-semibold text-gray-500 dark:text-gray-400 mb-2 px-1">Project Documents</h4>
          {isLoadingDocuments && <div className="text-center p-4"><Loader2 className="h-5 w-5 animate-spin inline-block"/></div>}
          {!isLoadingDocuments && documents.length === 0 && currentProjectId && (
            <p className="text-xs text-gray-500 dark:text-gray-400 p-2 text-center">No documents in this project.</p>
          )}
          {!currentProjectId && (
            <p className="text-xs text-gray-500 dark:text-gray-400 p-2 text-center">Select a project to see documents.</p>
          )}
          <div className="space-y-1">
            {documents.map(doc => (
              <Button
                key={doc.id}
                variant={currentDocument?.id === doc.id ? "secondary" : "ghost"}
                className="w-full justify-start text-left h-auto py-1.5 px-2"
                onClick={() => handleSelectDocument(doc.id)}
              >
                <FileText size={14} className="mr-2 flex-shrink-0" />
                <span className="text-sm truncate" title={doc.title}>{doc.title}</span>
              </Button>
            ))}
          </div>
        </ScrollArea>

        {/* Document Content Area */}
        <ScrollArea className="flex-1 p-4">
          {isLoadingContent && <div className="text-center p-10"><Loader2 className="h-8 w-8 animate-spin inline-block"/> <p>Loading content...</p></div>}
          {error && <p className="text-red-500 p-4">Error: {error}</p>}
          {!isLoadingContent && !error && currentDocumentContent && (
            <div className="prose dark:prose-invert max-w-none whitespace-pre-wrap">
              {currentDocumentContent}
            </div>
          )}
          {!isLoadingContent && !error && !currentDocumentContent && currentDocument && (
            <p className="text-gray-500 dark:text-gray-400 text-center p-10">Content not available or not loaded.</p>
          )}
          {!currentDocument && !isLoadingContent && !error && (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              <div
                className={`w-24 h-24 flex items-center justify-center rounded-full mb-6 ${
                  theme === "dark" ? "bg-gray-800" : "bg-gray-200"}
                }`}
              >
                <ListFilter size={40} className={theme === "dark" ? "text-gray-400" : "text-gray-500"} />
              </div>
              <h4 className="text-xl font-medium mb-3">No document selected</h4>
              <p className={`text-center text-sm max-w-md ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
                Select a document from the list on the left to view its content.
              </p>
            </div>
          )}
          {!isLoadingContent && !error && currentDocument && (
            <>
              {isEditing ? (
                <Textarea
                  value={editableContent}
                  onChange={handleContentChange}
                  rows={25}
                  className="mt-2 w-full flex-grow font-mono text-sm"
                  disabled={isSaving}
                />
              ) : (
                <ScrollArea className="mt-2 flex-grow whitespace-pre-wrap border rounded-md p-4 min-h-[300px] bg-white dark:bg-gray-800">
                  {currentDocumentContent || "Content not available or not loaded."}
                </ScrollArea>
              )}
              <div className="flex justify-between items-center mt-3 mb-3">
                {isEditing ? (
                    <Button onClick={handleSaveChanges} disabled={isSaving || !isEditing} variant="default" size="sm">
                        {isSaving ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Saving...</> : <><Save className="mr-2 h-4 w-4" /> Save Changes</>}
                    </Button>
                ) : (
                    <Button onClick={() => setIsEditing(true)} variant="outline" size="sm" disabled={isLoadingContent || isRefining}>
                        <FileText className="mr-2 h-4 w-4" /> Edit Document
                    </Button>
                )}
                {isEditing && (
                    <Button onClick={() => { setIsEditing(false); setEditableContent(currentDocumentContent || ""); }} variant="ghost" size="sm" disabled={isSaving}>
                        Cancel
                    </Button>
                )}
              </div>
              {!isEditing && (
                <div className="mt-4 pt-4 border-t dark:border-gray-700">
                  <h3 className="text-lg font-semibold mb-2">AI Refinement</h3>
                  <Textarea
                    placeholder="Tell the AI how to improve this document (e.g., make it more concise, add a section on X)..."
                    value={refinePrompt}
                    onChange={(e) => setRefinePrompt(e.target.value)}
                    rows={3}
                    className="mb-2"
                    disabled={isRefining || isLoadingContent}
                  />
                  <Button onClick={handleRefineDocument} disabled={isRefining || isLoadingContent || !refinePrompt.trim() || !currentDocument}>
                    {isRefining ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Refining...</> : <><FileText className="mr-2 h-4 w-4" /> Improve with AI</>}
                  </Button>
                </div>
              )}
            </>
          )}
        </ScrollArea>
      </div>
    </div>
  )
}
