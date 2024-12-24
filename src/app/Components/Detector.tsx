"use client";
import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { load as cocoSSDLoad, ObjectDetection as CocoSSD } from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import { renderPredictions } from "../../../utils/prediction";

let detectInterval: NodeJS.Timeout;

const ObjectDetection: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  async function runCoco() {
    setIsLoading(true);
    await tf.setBackend('webgl');
    await tf.ready();
    const net: CocoSSD = await cocoSSDLoad();
    setIsLoading(false);
    
    detectInterval = setInterval(() => {
      runObjectDetection(net);
    }, 10);
  }

  async function runObjectDetection(net: CocoSSD) {
    if (
      canvasRef.current &&
      webcamRef.current?.video &&
      webcamRef.current.video.readyState === 4
    ) {
      const videoEl = webcamRef.current.video;
      const canvas = canvasRef.current;
      
      // Get the container dimensions
      const containerRect = containerRef.current?.getBoundingClientRect();
      const videoRect = videoEl.getBoundingClientRect();
      
      if (!containerRect) return;

      // Calculate the actual displayed dimensions of the video
      const displayedWidth = videoRect.width;
      const displayedHeight = videoRect.height;

      // Set canvas dimensions to match the displayed video
      canvas.width = displayedWidth;
      canvas.height = displayedHeight;
      
      // Calculate offset to center the canvas over the video
      const offsetX = (containerRect.width - displayedWidth) / 2;
      const offsetY = (containerRect.height - displayedHeight) / 2;
      
      // Update canvas position
      canvas.style.left = `${offsetX}px`;
      canvas.style.top = `${offsetY}px`;
      canvas.style.width = `${displayedWidth}px`;
      canvas.style.height = `${displayedHeight}px`;

      const detectedObjects = await net.detect(
        videoEl,
        undefined,
        0.6
      );

      const context = canvas.getContext("2d");
      if (context) {
        // Calculate scale factors based on the actual displayed dimensions
        const scaleX = displayedWidth / videoEl.videoWidth;
        const scaleY = displayedHeight / videoEl.videoHeight;
        
        const scaledPredictions = detectedObjects.map(prediction => ({
          ...prediction,
          bbox: [
            prediction.bbox[0] * scaleX,
            prediction.bbox[1] * scaleY,
            prediction.bbox[2] * scaleX,
            prediction.bbox[3] * scaleY
          ] as [number, number, number, number]
        }));
        
        renderPredictions(scaledPredictions, context);
      }
    }
  }

  useEffect(() => {
    runCoco();
    return () => {
      if (detectInterval) {
        clearInterval(detectInterval);
      }
    };
  }, []);

  return (
    <div className="mt-8">
      {isLoading ? (
        <div className="gradient-text">Loading AI Model...</div>
      ) : (
        <div 
          ref={containerRef}
          className="relative flex justify-center items-center gradient p-1.5 rounded-md"
        >
          <Webcam
            ref={webcamRef}
            className="rounded-md w-[800px] lg:h-[600px] object-contain"
            muted
          />
          <canvas
            ref={canvasRef}
            className="absolute z-10 pointer-events-none w-[800px] lg:h-[600px]"
            style={{ position: 'absolute' }}
          />
        </div>
      )}
    </div>
  );
};

export default ObjectDetection;