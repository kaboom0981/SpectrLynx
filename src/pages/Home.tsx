import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Target, Eye, Radar, ImageIcon } from "lucide-react";

const Home = () => {
  return (
    <div className="min-h-screen bg-background">
      <section className="relative py-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-military-dark/20 via-background to-military-green/10" />
        <div className="container mx-auto max-w-6xl relative z-10">
          <div className="text-center space-y-6">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
              <Target className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-primary">AUTO-AD Detection System</span>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold text-foreground tracking-tight">
              Hyperspectral Anomaly
              <br />
              <span className="text-primary">Detection Platform</span>
            </h1>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Advanced autoencoder-based system for detecting camouflaged objects, animals, vehicles, and anomalies in hyperspectral imagery.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link to="/detect">
                <Button size="lg" className="gap-2">
                  <Radar className="h-5 w-5" />
                  Start Detection
                </Button>
              </Link>
              <Link to="/about">
                <Button size="lg" variant="outline" className="gap-2">
                  <Eye className="h-5 w-5" />
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 px-4 bg-card/30">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center mb-12 text-foreground">
            Detection Capabilities
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="border-border bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Target className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">Multi-Format Support</h3>
                <p className="text-muted-foreground">
                  Process CSV, NPY, JPG, JPEG, and PNG files with automatic format conversion and preprocessing.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Radar className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">Real-Time Processing</h3>
                <p className="text-muted-foreground">
                  Advanced autoencoder architecture delivers fast, accurate anomaly detection with visual feedback.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <ImageIcon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">Rich Visualizations</h3>
                <p className="text-muted-foreground">
                  Generate heatmaps, spectral plots, ROC curves, 3D cubes, and AI-powered anomaly descriptions.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="py-16 px-4">
        <div className="container mx-auto max-w-4xl">
          <div className="bg-gradient-to-r from-primary/10 to-accent/10 rounded-lg p-8 md:p-12 border border-primary/20">
            <h2 className="text-3xl font-bold mb-4 text-foreground">
              Ready to detect anomalies?
            </h2>
            <p className="text-lg text-muted-foreground mb-6">
              Upload your hyperspectral data and get comprehensive analysis with multiple visualization outputs.
            </p>
            <Link to="/detect">
              <Button size="lg" className="gap-2">
                <Target className="h-5 w-5" />
                Start Detection Now
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
