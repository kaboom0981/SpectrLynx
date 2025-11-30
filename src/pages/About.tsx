import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle } from "lucide-react";

const About = () => {
  const comparisonData = [
    {
      method: "AUTO-AD (Autoencoder)",
      unsupervised: true,
      realTime: true,
      accuracy: "High",
      complexity: "Medium",
      highlights: "Symmetric architecture, reconstruction error-based",
    },
    {
      method: "GAN-based Detection",
      unsupervised: true,
      realTime: false,
      accuracy: "High",
      complexity: "High",
      highlights: "Adversarial training, requires large datasets",
    },
    {
      method: "CNN Classifiers",
      unsupervised: false,
      realTime: true,
      accuracy: "Very High",
      complexity: "Medium",
      highlights: "Requires labeled data, supervised learning",
    },
    {
      method: "Hybrid Methods",
      unsupervised: true,
      realTime: false,
      accuracy: "Very High",
      complexity: "Very High",
      highlights: "Combines multiple approaches, computationally intensive",
    },
  ];

  return (
    <div className="min-h-screen bg-background py-12 px-4">
      <div className="container mx-auto max-w-6xl space-y-12">
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground">
            About AUTO-AD
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Advanced hyperspectral anomaly detection using autoencoder-based deep learning
          </p>
        </div>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-2xl text-foreground">What is Hyperspectral Anomaly Detection?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-muted-foreground">
            <p>
              Hyperspectral imaging captures data across hundreds of narrow spectral bands, revealing information invisible to the human eye. 
              This technology enables the detection of camouflaged objects, concealed materials, and subtle anomalies in complex environments.
            </p>
            <p>
              Traditional RGB cameras capture only three color channels (red, green, blue), while hyperspectral sensors can capture 
              data across 100+ wavelength bands, providing a unique "spectral signature" for every pixel in the image.
            </p>
            <p>
              Applications include military surveillance, environmental monitoring, agricultural analysis, medical diagnostics, 
              and mineral exploration. The ability to detect camouflaged objects makes this technology particularly valuable for 
              defense and security operations.
            </p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-2xl text-foreground">The AUTO-AD Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-muted-foreground">
            <p>
              AUTO-AD (Autoencoder-based Anomaly Detection) employs a symmetric deep learning architecture to identify anomalies 
              through reconstruction error analysis. The model learns normal patterns in hyperspectral data and flags deviations as potential anomalies.
            </p>
            
            <div className="bg-muted/30 rounded-lg p-6 space-y-3">
              <h4 className="font-semibold text-foreground text-lg">Architecture Overview</h4>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-foreground mb-2">Encoder Path</h5>
                  <ul className="space-y-1 text-sm">
                    <li>• Input Layer: 200 spectral bands</li>
                    <li>• Hidden Layer 1: 64 neurons</li>
                    <li>• Bottleneck: 16 neurons (compressed representation)</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-foreground mb-2">Decoder Path</h5>
                  <ul className="space-y-1 text-sm">
                    <li>• Hidden Layer 1: 64 neurons</li>
                    <li>• Output Layer: 200 spectral bands (reconstruction)</li>
                    <li>• Anomaly Score: Reconstruction error magnitude</li>
                  </ul>
                </div>
              </div>
            </div>

            <p>
              The model calculates reconstruction error for each pixel. High reconstruction error indicates the pixel's spectral 
              signature deviates from learned normal patterns, suggesting the presence of an anomaly such as a camouflaged object, 
              vehicle, or unusual material.
            </p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-2xl text-foreground">Method Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-foreground">Method</TableHead>
                    <TableHead className="text-foreground text-center">Unsupervised</TableHead>
                    <TableHead className="text-foreground text-center">Real-Time</TableHead>
                    <TableHead className="text-foreground">Accuracy</TableHead>
                    <TableHead className="text-foreground">Complexity</TableHead>
                    <TableHead className="text-foreground">Key Features</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {comparisonData.map((row) => (
                    <TableRow key={row.method}>
                      <TableCell className="font-medium text-foreground">
                        {row.method === "AUTO-AD (Autoencoder)" ? (
                          <Badge className="bg-primary text-primary-foreground">{row.method}</Badge>
                        ) : (
                          row.method
                        )}
                      </TableCell>
                      <TableCell className="text-center">
                        {row.unsupervised ? (
                          <CheckCircle2 className="h-5 w-5 text-primary mx-auto" />
                        ) : (
                          <XCircle className="h-5 w-5 text-muted-foreground mx-auto" />
                        )}
                      </TableCell>
                      <TableCell className="text-center">
                        {row.realTime ? (
                          <CheckCircle2 className="h-5 w-5 text-primary mx-auto" />
                        ) : (
                          <XCircle className="h-5 w-5 text-muted-foreground mx-auto" />
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground">{row.accuracy}</TableCell>
                      <TableCell className="text-muted-foreground">{row.complexity}</TableCell>
                      <TableCell className="text-muted-foreground text-sm">{row.highlights}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              * AUTO-AD combines unsupervised learning with real-time processing capabilities, making it ideal for operational deployment 
              where labeled training data is scarce or unavailable.
            </p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-2xl text-foreground">Output Visualizations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-muted-foreground">
            <p>The system generates comprehensive analysis outputs including:</p>
            <ul className="space-y-2 ml-6">
              <li><strong className="text-foreground">Heatmap:</strong> Color-coded anomaly score visualization showing detection confidence</li>
              <li><strong className="text-foreground">Binary Mask:</strong> Clear identification of anomalous pixels (1 = anomaly, 0 = normal)</li>
              <li><strong className="text-foreground">Spectral Signatures:</strong> Wavelength plots for selected pixels showing unique spectral patterns</li>
              <li><strong className="text-foreground">ROC Curve:</strong> Receiver Operating Characteristic for model performance evaluation</li>
              <li><strong className="text-foreground">3D Hyperspectral Cube:</strong> Interactive 3D visualization of the complete data cube</li>
              <li><strong className="text-foreground">AI Description:</strong> Automated text analysis identifying detected anomaly types</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default About;
