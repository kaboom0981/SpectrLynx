import { useState } from "react";
import { useLocation, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Download, Mail, FileText, ArrowLeft } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Results = () => {
  const location = useLocation();
  const { toast } = useToast();
  const fileName = location.state?.fileName || "unknown.jpg";
  const [selectedPixel, setSelectedPixel] = useState<{ x: number; y: number } | null>(null);

  const handleDownload = (type: string) => {
    toast({
      title: "Download started",
      description: `Downloading ${type}...`,
    });
  };

  const handleEmailResults = () => {
    toast({
      title: "Email feature",
      description: "Navigate to Contact page to email results.",
    });
  };

  return (
    <div className="min-h-screen bg-background py-12 px-4">
      <div className="container mx-auto max-w-7xl space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <Link to="/detect" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
              Back to Detection
            </Link>
            <h1 className="text-4xl font-bold text-foreground">Detection Results</h1>
            <p className="text-muted-foreground">Analysis for: {fileName}</p>
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleEmailResults} className="gap-2">
              <Mail className="h-4 w-4" />
              Email Results
            </Button>
            <Button onClick={() => handleDownload("PDF Report")} className="gap-2">
              <FileText className="h-4 w-4" />
              Export PDF
            </Button>
          </div>
        </div>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground">AI Analysis Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="text-foreground">
                <strong>Detected Anomaly Type:</strong> Camouflaged vehicle
              </p>
              <p className="text-muted-foreground mt-2">
                Analysis indicates the presence of a military vehicle with spectral signatures consistent 
                with camouflage materials. High anomaly scores detected in the central-right region of the image, 
                with distinctive spectral patterns in the near-infrared range suggesting painted metal surfaces 
                beneath vegetation-like covering.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground">Visualizations</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="heatmap" className="w-full">
              <TabsList className="grid grid-cols-6 w-full">
                <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
                <TabsTrigger value="mask">Binary Mask</TabsTrigger>
                <TabsTrigger value="spectral">Spectral Plot</TabsTrigger>
                <TabsTrigger value="roc">ROC Curve</TabsTrigger>
                <TabsTrigger value="cube">3D Cube</TabsTrigger>
                <TabsTrigger value="pixel">Pixel Tool</TabsTrigger>
              </TabsList>

              <TabsContent value="heatmap" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="text-6xl">🔥</div>
                    <p className="text-foreground font-medium">Anomaly Heatmap</p>
                    <p className="text-sm text-muted-foreground">
                      Color-coded visualization showing anomaly detection confidence
                    </p>
                  </div>
                </div>
                <Button onClick={() => handleDownload("heatmap")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download Heatmap
                </Button>
              </TabsContent>

              <TabsContent value="mask" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="text-6xl">⬛⬜</div>
                    <p className="text-foreground font-medium">Binary Mask</p>
                    <p className="text-sm text-muted-foreground">
                      1 = Anomaly detected, 0 = Normal region
                    </p>
                  </div>
                </div>
                <Button onClick={() => handleDownload("binary mask")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download Binary Mask
                </Button>
              </TabsContent>

              <TabsContent value="spectral" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="text-6xl">📊</div>
                    <p className="text-foreground font-medium">Spectral Signature Plot</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedPixel
                        ? `Showing spectrum for pixel (${selectedPixel.x}, ${selectedPixel.y})`
                        : "Select a pixel using the Pixel Tool tab"}
                    </p>
                  </div>
                </div>
                <Button onClick={() => handleDownload("spectral plot")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download Spectral Plot
                </Button>
              </TabsContent>

              <TabsContent value="roc" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="text-6xl">📈</div>
                    <p className="text-foreground font-medium">ROC Curve</p>
                    <p className="text-sm text-muted-foreground">
                      Model performance evaluation - AUC: 0.94
                    </p>
                  </div>
                </div>
                <Button onClick={() => handleDownload("ROC curve")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download ROC Curve
                </Button>
              </TabsContent>

              <TabsContent value="cube" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="text-6xl">🧊</div>
                    <p className="text-foreground font-medium">3D Hyperspectral Cube</p>
                    <p className="text-sm text-muted-foreground">
                      Interactive 3D visualization of spectral data cube
                    </p>
                  </div>
                </div>
                <Button onClick={() => handleDownload("3D cube")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download 3D Visualization
                </Button>
              </TabsContent>

              <TabsContent value="pixel" className="space-y-4">
                <div className="bg-muted/30 rounded-lg p-8 min-h-96">
                  <div className="text-center space-y-4">
                    <p className="text-foreground font-medium">Pixel Selection Tool</p>
                    <p className="text-sm text-muted-foreground mb-4">
                      Click on the image to select a pixel and view its spectral signature
                    </p>
                    <div className="inline-block bg-background rounded border border-border cursor-crosshair">
                      <div
                        className="w-96 h-96 bg-gradient-to-br from-military-dark via-military-green to-military-olive"
                        onClick={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          const x = Math.floor(((e.clientX - rect.left) / rect.width) * 100);
                          const y = Math.floor(((e.clientY - rect.top) / rect.height) * 100);
                          setSelectedPixel({ x, y });
                          toast({
                            title: "Pixel selected",
                            description: `Coordinates: (${x}, ${y})`,
                          });
                        }}
                      />
                    </div>
                    {selectedPixel && (
                      <div className="mt-4 p-4 bg-primary/5 border border-primary/20 rounded-lg">
                        <p className="text-foreground">
                          Selected Pixel: ({selectedPixel.x}, {selectedPixel.y})
                        </p>
                        <p className="text-sm text-muted-foreground">
                          View spectral signature in the "Spectral Plot" tab
                        </p>
                      </div>
                    )}
                  </div>
                </div>
                <Button onClick={() => handleDownload("pixel data CSV")} className="gap-2">
                  <Download className="h-4 w-4" />
                  Export Anomaly Pixels (CSV)
                </Button>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Results;
