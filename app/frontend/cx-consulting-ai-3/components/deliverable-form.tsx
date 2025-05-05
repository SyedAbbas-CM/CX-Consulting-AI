import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Upload, FileText, Info } from "lucide-react"

interface DeliverableFormProps {
  theme: "light" | "dark"
}

export default function DeliverableForm({ theme }: DeliverableFormProps) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Generate Deliverable</h2>
      <p className={`mb-6 ${theme === "dark" ? "text-gray-400" : "text-gray-600"}`}>
        Fill in the details to generate a professional CX consulting deliverable.
      </p>

      <Tabs defaultValue="basic" className="w-full">
        <TabsList className={`grid w-full grid-cols-3 ${theme === "dark" ? "bg-gray-800" : "bg-gray-100"}`}>
          <TabsTrigger value="basic">Basic Info</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
          <TabsTrigger value="context">Context Files</TabsTrigger>
        </TabsList>

        {/* Basic Info Tab */}
        <TabsContent value="basic" className="mt-6 space-y-6">
          <div className="space-y-4">
            <div>
              <label className="block mb-2 text-sm font-medium">
                Deliverable Type <span className="text-red-500">*</span>
              </label>
              <Select>
                <SelectTrigger className={theme === "dark" ? "bg-gray-800 border-gray-700" : ""}>
                  <SelectValue placeholder="Select deliverable type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cx-strategy">CX Strategy</SelectItem>
                  <SelectItem value="journey-map">Customer Journey Map</SelectItem>
                  <SelectItem value="ux-audit">UX Audit</SelectItem>
                  <SelectItem value="service-blueprint">Service Blueprint</SelectItem>
                  <SelectItem value="research-report">Research Report</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">
                Client Name <span className="text-red-500">*</span>
              </label>
              <Input
                placeholder="Enter client name"
                className={theme === "dark" ? "bg-gray-800 border-gray-700" : ""}
              />
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">
                Project Requirements <span className="text-red-500">*</span>
              </label>
              <Textarea
                placeholder="Describe the project requirements and goals"
                className={`min-h-[100px] ${theme === "dark" ? "bg-gray-800 border-gray-700" : ""}`}
              />
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">Additional Context</label>
              <Textarea
                placeholder="Any additional information or context that might be helpful"
                className={`min-h-[80px] ${theme === "dark" ? "bg-gray-800 border-gray-700" : ""}`}
              />
            </div>
          </div>
        </TabsContent>

        {/* Advanced Tab */}
        <TabsContent value="advanced" className="mt-6 space-y-6">
          <div className="space-y-4">
            <div>
              <label className="block mb-2 text-sm font-medium">Industry</label>
              <Select>
                <SelectTrigger className={theme === "dark" ? "bg-gray-800 border-gray-700" : ""}>
                  <SelectValue placeholder="Select industry" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="technology">Technology</SelectItem>
                  <SelectItem value="finance">Finance</SelectItem>
                  <SelectItem value="healthcare">Healthcare</SelectItem>
                  <SelectItem value="retail">Retail</SelectItem>
                  <SelectItem value="education">Education</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">Target Audience</label>
              <Select>
                <SelectTrigger className={theme === "dark" ? "bg-gray-800 border-gray-700" : ""}>
                  <SelectValue placeholder="Select target audience" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="executives">Executives</SelectItem>
                  <SelectItem value="product-teams">Product Teams</SelectItem>
                  <SelectItem value="marketing">Marketing</SelectItem>
                  <SelectItem value="customer-service">Customer Service</SelectItem>
                  <SelectItem value="technical">Technical Teams</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">Document Format Preferences</label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div
                  className={`flex items-center space-x-3 p-4 border rounded-md ${theme === "dark" ? "border-gray-700 bg-gray-800/50" : "border-gray-200"}`}
                >
                  <Checkbox id="executive-summary" />
                  <label htmlFor="executive-summary" className="text-sm font-medium flex items-center">
                    <FileText size={16} className="mr-2" />
                    Include Executive Summary
                  </label>
                </div>
                <div
                  className={`flex items-center space-x-3 p-4 border rounded-md ${theme === "dark" ? "border-gray-700 bg-gray-800/50" : "border-gray-200"}`}
                >
                  <Checkbox id="case-studies" />
                  <label htmlFor="case-studies" className="text-sm font-medium flex items-center">
                    <FileText size={16} className="mr-2" />
                    Include Case Studies
                  </label>
                </div>
              </div>
            </div>

            <div
              className={`p-4 border rounded-md ${theme === "dark" ? "border-gray-700 bg-gray-800/50" : "border-gray-200"}`}
            >
              <div className="flex items-start space-x-3">
                <Info size={20} className="mt-0.5 text-blue-500" />
                <div>
                  <h4 className="text-sm font-medium mb-1">Advanced Options</h4>
                  <p className={`text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-600"}`}>
                    These settings help tailor the document to your specific audience and industry context.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Context Files Tab */}
        <TabsContent value="context" className="mt-6">
          <div
            className={`border-2 border-dashed rounded-lg p-10 text-center ${theme === "dark" ? "border-gray-700" : "border-gray-300"}`}
          >
            <div className="flex flex-col items-center justify-center">
              <Upload size={40} className={`mb-4 ${theme === "dark" ? "text-gray-500" : "text-gray-400"}`} />
              <h3 className="text-lg font-medium mb-2">Drag & drop files here</h3>
              <p className={`mb-4 text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-500"}`}>
                or click to browse files
              </p>
              <p className={`text-xs ${theme === "dark" ? "text-gray-500" : "text-gray-400"}`}>
                Supports PDF, Word, and text files
              </p>
            </div>
          </div>

          <div
            className={`mt-6 p-4 border rounded-md ${theme === "dark" ? "border-gray-700 bg-gray-800/50" : "border-gray-200"}`}
          >
            <div className="flex items-start space-x-3">
              <Upload size={20} className="mt-0.5" />
              <div>
                <h4 className="text-sm font-medium mb-1">Context Files</h4>
                <p className={`text-sm ${theme === "dark" ? "text-gray-400" : "text-gray-600"}`}>
                  Upload additional documents to provide context for the AI. This can include existing materials,
                  research, or reference documents.
                </p>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      <div className="mt-8 flex justify-end">
        <Button
          className={`px-6 ${theme === "dark" ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600"}`}
        >
          Generate Document
        </Button>
      </div>
    </div>
  )
}
