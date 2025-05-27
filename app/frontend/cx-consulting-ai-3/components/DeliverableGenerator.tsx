'use client';

import React from 'react';
import { Button } from "@/components/ui/button";
import { useChatStore } from '@/store/chatStore'; // Adjusted import path
import type { DocumentGenerationConfig } from '@/src/types/api'; // Keep type for deliverable keys

// Define the available deliverable types and their user-friendly labels
const DELIVERABLE_TYPES: Array<{ key: DocumentGenerationConfig['deliverable_type']; label: string }> = [
  { key: "cx_strategy", label: "CX Strategy" },
  { key: "roi_analysis", label: "ROI Analysis" },
  { key: "journey_map", label: "Journey Map" },
  { key: "proposal", label: "Proposal" },
  // Add other deliverable types here if needed
];

// Interface for individual button props (derived from the plan snippet)
interface DeliverableButtonProps {
  type: DocumentGenerationConfig['deliverable_type'];
  label: string;
}

const DeliverableButton: React.FC<DeliverableButtonProps> = ({ type, label }) => {
  const activeType = useChatStore(state => state.activeDeliverableType);
  const setActiveDeliverableType = useChatStore(state => state.setActiveDeliverableType);

  const isActive = activeType === type;

  return (
    <Button
      variant={isActive ? "secondary" : "outline"} // "secondary" for active, "outline" for inactive
      data-active={isActive} // For potential CSS styling via data attribute
      onClick={() => setActiveDeliverableType(isActive ? null : type)} // Toggle: set to type, or null if already active
      className="flex-grow sm:flex-grow-0" // Allow buttons to grow on small screens
    >
      {label}
    </Button>
  );
};

export function DeliverableGenerator() {
  // This component no longer handles generation itself, only selection.
  // Props like currentProjectIdFromChat, onGenerationComplete are no longer needed here.

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Activate a Deliverable:
      </p>
      <div className="flex flex-wrap gap-2">
        {DELIVERABLE_TYPES.map(({ key, label }) => (
          <DeliverableButton key={key} type={key} label={label} />
        ))}
      </div>
    </div>
  );
}
