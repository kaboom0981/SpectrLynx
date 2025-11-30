import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileImage, Radar, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Detect = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "text/csv", "application/octet-stream"];
    const fileExt = selectedFile.name.split(".").pop()?.toLowerCase();
    
    if (!validTypes.includes(selectedFile.type) && !["csv", "npy", "jpg", "jpeg", "png"].includes(fileExt || "")) {
      toast({
        title: "Invalid file type",
        description: "Please upload JPG, JPEG, PNG, CSV, or NPY files only.",
        variant: "destructive",
      });
      return;
    }

    setFile(selectedFile);

    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }

    toast({
      title: "File loaded",
      description: `${selectedFile.name} ready for detection`,
    });
  };

  const handleRemoveFile = () => {
    setFile(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleStartDetection = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please upload a file before starting detection.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);

    // Simulate processing time
    setTimeout(() => {
      setIsProcessing(false);
      toast({
        title: "Detection complete",
        description: "Anomaly analysis finished successfully.",
      });
      navigate("/results", { state: { fileName: file.name } });
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-background py-12 px-4">
      <div className="container mx-auto max-w-4xl space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-foreground">Anomaly Detection</h1>
          <p className="text-lg text-muted-foreground">
            Upload hyperspectral data or images for analysis
          </p>
        </div>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground">Upload File</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${
                file
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/30"
              }`}
              onClick={() => !file && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.csv,.npy"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              {file ? (
                <div className="space-y-4">
                  <FileImage className="h-16 w-16 text-primary mx-auto" />
                  <div className="space-y-2">
                    <p className="font-medium text-foreground">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile();
                    }}
                    className="gap-2"
                  >
                    <X className="h-4 w-4" />
                    Remove
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-16 w-16 text-muted-foreground mx-auto" />
                  <div className="space-y-2">
                    <p className="text-foreground font-medium">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-muted-foreground">
                      JPG, JPEG, PNG, CSV, or NPY files
                    </p>
                  </div>
                </div>
              )}
            </div>

            {preview && (
              <div className="space-y-4">
                <h3 className="font-semibold text-foreground">Preview</h3>
                <div className="rounded-lg overflow-hidden border border-border bg-muted/20">
                  <img
                    src={preview}
                    alt="File preview"
                    className="w-full h-auto max-h-96 object-contain"
                  />
                </div>
              </div>
            )}

            <Button
              onClick={handleStartDetection}
              disabled={!file || isProcessing}
              size="lg"
              className="w-full gap-2"
            >
              <Radar className="h-5 w-5" />
              {isProcessing ? "Processing..." : "Start Detection"}
            </Button>
          </CardContent>
        </Card>

        {isProcessing && (
          <Card className="border-primary bg-card">
            <CardContent className="p-8">
              <div className="flex flex-col items-center justify-center space-y-6">
                <div className="relative w-32 h-32">
                  <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
                  <div className="absolute inset-0 rounded-full border-4 border-t-primary animate-radar-sweep" />
                  <Radar className="absolute inset-0 m-auto h-12 w-12 text-primary animate-pulse-glow" />
                </div>
                <div className="text-center space-y-2">
                  <p className="text-lg font-semibold text-foreground">
                    Processing hyperspectral data...
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Analyzing spectral signatures and detecting anomalies
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Detect;
