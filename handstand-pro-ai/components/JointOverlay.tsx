
import React from 'react';
import { HandstandAnalysis, Point } from '../types';

interface JointOverlayProps {
  analysis: HandstandAnalysis;
}

const JointOverlay: React.FC<JointOverlayProps> = ({ analysis }) => {
  const { leftSide, rightSide, torque } = analysis;

  const drawLine = (p1: Point, p2: Point, color: string, dashed = false) => (
    <line 
      x1={`${p1.x}%`} y1={`${p1.y}%`} 
      x2={`${p2.x}%`} y2={`${p2.y}%`} 
      stroke={color} 
      strokeWidth="2.5" 
      strokeLinecap="round"
      strokeDasharray={dashed ? "5,5" : "none"}
    />
  );

  const drawJoint = (p: Point, color: string, size = 5) => (
    <circle cx={`${p.x}%`} cy={`${p.y}%`} r={size} fill={color} stroke="white" strokeWidth="1.5" />
  );

  const renderSide = (side: any, color: string) => (
    <g key={color} opacity="0.8">
      {drawLine(side.wrist, side.elbow, color)}
      {drawLine(side.elbow, side.shoulder, color)}
      {drawLine(side.shoulder, side.hip, color)}
      {drawLine(side.hip, side.knee, color)}
      {drawLine(side.knee, side.ankle, color)}
      
      {drawJoint(side.wrist, color)}
      {drawJoint(side.elbow, color)}
      {drawJoint(side.shoulder, color)}
      {drawJoint(side.hip, color)}
      {drawJoint(side.knee, color)}
      {drawJoint(side.ankle, color)}
    </g>
  );

  const getMidPoint = (p1: Point, p2: Point): Point => ({
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2
  });

  const shoulderMid = getMidPoint(leftSide.shoulder, rightSide.shoulder);
  const hipMid = getMidPoint(leftSide.hip, rightSide.hip);
  const ankleMid = getMidPoint(leftSide.ankle, rightSide.ankle);

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none drop-shadow-md">
      {/* Plumb Line (Balance Line) */}
      <line 
        x1={`${torque.balanceLineX}%`} y1="0%" 
        x2={`${torque.balanceLineX}%`} y2="100%" 
        stroke="rgba(0,0,0,0.2)" 
        strokeWidth="1" 
        strokeDasharray="4,4"
      />

      {/* Torque Offsets (Leverage indicators) */}
      {torque.offsets.shoulder !== 0 && (
        drawLine({ x: torque.balanceLineX, y: shoulderMid.y }, shoulderMid, "#f59e0b", true)
      )}
      {torque.offsets.hip !== 0 && (
        drawLine({ x: torque.balanceLineX, y: hipMid.y }, hipMid, "#f59e0b", true)
      )}
      {torque.offsets.ankle !== 0 && (
        drawLine({ x: torque.balanceLineX, y: ankleMid.y }, ankleMid, "#f59e0b", true)
      )}

      {/* Joint Segments */}
      {renderSide(leftSide, '#2563eb')}
      {renderSide(rightSide, '#dc2626')}

      {/* Center of Mass Marker */}
      <g>
        <circle 
          cx={`${torque.centerOfMass.x}%`} 
          cy={`${torque.centerOfMass.y}%`} 
          r="8" 
          fill="none" 
          stroke="#10b981" 
          strokeWidth="2" 
        />
        <line 
          x1={`${torque.centerOfMass.x - 2}%`} y1={`${torque.centerOfMass.y}%`} 
          x2={`${torque.centerOfMass.x + 2}%`} y2={`${torque.centerOfMass.y}%`} 
          stroke="#10b981" strokeWidth="2" 
        />
        <line 
          x1={`${torque.centerOfMass.x}%`} y1={`${torque.centerOfMass.y - 2}%`} 
          x2={`${torque.centerOfMass.x}%`} y2={`${torque.centerOfMass.y + 2}%`} 
          stroke="#10b981" strokeWidth="2" 
        />
      </g>
    </svg>
  );
};

export default JointOverlay;
