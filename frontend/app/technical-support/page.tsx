"use client"

import { useState, useEffect } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { Header } from "@/components/header"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import {
    Headphones,
    AlertCircle,
    CheckCircle2,
    Clock,
    Upload,
    Monitor,
    Smartphone,
    Camera,
    Wifi,
    Database,
    Shield,
    PhoneCall,
    Mic,
} from "lucide-react"
import { supportTicketSchema, type SupportTicketInput } from "@/lib/validation"

const issueCategories = [
    { value: "image-upload", label: "Image Upload Issues", icon: Upload },
    { value: "analysis", label: "Analysis Problems", icon: Camera },
    { value: "account", label: "Account & Login", icon: Shield },
    { value: "display", label: "Display Issues", icon: Monitor },
    { value: "mobile", label: "Mobile App Issues", icon: Smartphone },
    { value: "connectivity", label: "Connectivity Issues", icon: Wifi },
    { value: "data", label: "Data & Reports", icon: Database },
    { value: "other", label: "Other", icon: AlertCircle },
]

const faqItems = [
    {
        question: "My tire image upload is failing. What should I do?",
        answer: "First, ensure your image is in JPG, PNG, or WebP format and under 10MB. Make sure you have a stable internet connection. If the issue persists, try clearing your browser cache or using a different browser. For mobile uploads, check that the app has camera and storage permissions."
    },
    {
        question: "The analysis is taking longer than expected.",
        answer: "Analysis typically completes within 10-15 seconds. If it takes longer, check your internet connection. During peak hours, there might be slight delays. If the analysis fails, try uploading a clearer image with better lighting and ensure the entire tire tread is visible."
    },
    {
        question: "I cannot access my previous analysis history.",
        answer: "Make sure you are logged into the same account used for previous analyses. History is account-specific and stored for 12 months. If you have recently changed your email, contact support to link your history to your new account."
    },
    {
        question: "The wear pattern detection seems inaccurate.",
        answer: "For best results, photograph the tire from directly above with even lighting. Avoid shadows and ensure the tread pattern is clean and visible. If you believe there is a detection error, submit a support ticket with the image for our team to review."
    },
    {
        question: "How do I export my analysis reports?",
        answer: "Navigate to your History page, select the analysis you want to export, and click the Export button. Reports are available in PDF and CSV formats. Professional and Enterprise users can batch export multiple reports at once."
    },
    {
        question: "The app is not working on my mobile device.",
        answer: "Ensure your app is updated to the latest version. Check that you have granted necessary permissions (camera, storage, internet). Try force-closing and reopening the app. If issues persist, uninstall and reinstall the application."
    },
]

const statusItems = [
    { service: "API Services", status: "operational", uptime: "99.98%" },
    { service: "Image Processing", status: "operational", uptime: "99.95%" },
    { service: "User Authentication", status: "operational", uptime: "99.99%" },
    { service: "Report Generation", status: "operational", uptime: "99.97%" },
]

export default function TechnicalSupportPage() {
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [isSubmitted, setIsSubmitted] = useState(false)
    const [ticketNumber, setTicketNumber] = useState<string | null>(null)
    const [voiceAgentStatus, setVoiceAgentStatus] = useState<string>("checking")
    const [voiceAgentName, setVoiceAgentName] = useState<string>("")
    const [isInitializing, setIsInitializing] = useState(false)

    useEffect(() => {
        const checkVoiceAgent = async () => {
            try {
                const res = await fetch("http://localhost:8000/support/voice-agent")
                const data = await res.json()
                if (data.status === "active") {
                    setVoiceAgentStatus("active")
                    setVoiceAgentName(data.name || "Smart Tire Analyser Support")
                } else if (data.status === "not_initialized") {
                    setVoiceAgentStatus("ready")
                } else {
                    setVoiceAgentStatus("unavailable")
                }
            } catch {
                setVoiceAgentStatus("unavailable")
            }
        }
        checkVoiceAgent()
    }, [])

    const handleInitVoiceAgent = async () => {
        setIsInitializing(true)
        try {
            const res = await fetch("http://localhost:8000/support/voice-agent/init", { method: "POST" })
            const data = await res.json()
            if (data.status === "active") {
                setVoiceAgentStatus("active")
                setVoiceAgentName(data.name || "Smart Tire Analyser Support")
            } else {
                setVoiceAgentStatus("error")
            }
        } catch {
            setVoiceAgentStatus("error")
        } finally {
            setIsInitializing(false)
        }
    }

    const {
        register,
        handleSubmit,
        formState: { errors },
        setValue,
        reset,
    } = useForm<SupportTicketInput>({
        resolver: zodResolver(supportTicketSchema),
    })

    const onSubmit = async (data: SupportTicketInput) => {
        setIsSubmitting(true)
        await new Promise(resolve => setTimeout(resolve, 1500))
        setTicketNumber(`ST-${Math.floor(100000 + Math.random() * 900000)}`)
        setIsSubmitting(false)
        setIsSubmitted(true)
    }

    return (
        <div className="flex min-h-screen flex-col">
            <Header />
            <main className="flex-1 pt-16">
                {/* Hero Section */}
                <section className="border-b border-border/50 py-12">
                    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                        <div className="flex items-center gap-4">
                            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                                <Headphones className="h-7 w-7 text-primary" />
                            </div>
                            <div>
                                <h1 className="text-3xl font-bold tracking-tight text-foreground">
                                    Technical Support
                                </h1>
                                <p className="text-muted-foreground">
                                    Get help with platform issues and technical problems
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* System Status */}
                <section className="border-b border-border/50 py-8">
                    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <CheckCircle2 className="h-5 w-5 text-green-500" />
                                <span className="font-medium text-foreground">All Systems Operational</span>
                            </div>
                            <div className="flex gap-6">
                                {statusItems.map((item) => (
                                    <div key={item.service} className="hidden text-sm sm:flex sm:items-center sm:gap-2">
                                        <span className="h-2 w-2 rounded-full bg-green-500" />
                                        <span className="text-muted-foreground">{item.service}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* Main Content */}
                <section className="py-12">
                    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                        <div className="grid gap-12 lg:grid-cols-2">
                            {/* Support Ticket Form */}
                            <div>
                                <h2 className="text-2xl font-bold text-foreground">Submit a Support Ticket</h2>
                                <p className="mt-2 text-muted-foreground">
                                    Describe your issue and our technical team will respond within 24 hours.
                                </p>

                                <Card className="mt-6 border-border/50 bg-card/50">
                                    <CardContent className="pt-6">
                                        {isSubmitted ? (
                                            <div className="flex flex-col items-center justify-center py-12 text-center">
                                                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-500/20">
                                                    <CheckCircle2 className="h-8 w-8 text-green-500" />
                                                </div>
                                                <h3 className="mt-6 text-lg font-semibold text-foreground">
                                                    Ticket Submitted!
                                                </h3>
                                                <p className="mt-2 text-muted-foreground">
                                                    Ticket #{ticketNumber} has been created.
                                                    We will respond within 24 hours.
                                                </p>
                                                <Button
                                                    variant="outline"
                                                    className="mt-6"
                                                    onClick={() => {
                                                        setTicketNumber(null)
                                                        setIsSubmitted(false)
                                                        reset()
                                                    }}
                                                >
                                                    Submit Another Ticket
                                                </Button>
                                            </div>
                                        ) : (
                                            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                                                <div className="space-y-2">
                                                    <Label htmlFor="category">Issue Category</Label>
                                                    <Select onValueChange={(value) => setValue("category", value)}>
                                                        <SelectTrigger>
                                                            <SelectValue placeholder="Select issue type" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            {issueCategories.map((cat) => (
                                                                <SelectItem key={cat.value} value={cat.value}>
                                                                    <div className="flex items-center gap-2">
                                                                        <cat.icon className="h-4 w-4" />
                                                                        {cat.label}
                                                                    </div>
                                                                </SelectItem>
                                                            ))}
                                                        </SelectContent>
                                                    </Select>
                                                    {errors.category && (
                                                        <p className="text-sm text-destructive">{errors.category.message}</p>
                                                    )}
                                                </div>

                                                <div className="grid gap-4 sm:grid-cols-2">
                                                    <div className="space-y-2">
                                                        <Label htmlFor="email">Email Address</Label>
                                                        <Input id="email" type="email" placeholder="you@example.com" {...register("email")} />
                                                        {errors.email && (
                                                            <p className="text-sm text-destructive">{errors.email.message}</p>
                                                        )}
                                                    </div>
                                                    <div className="space-y-2">
                                                        <Label htmlFor="priority">Priority</Label>
                                                        <Select onValueChange={(value) => setValue("priority", value as SupportTicketInput["priority"])}>
                                                            <SelectTrigger>
                                                                <SelectValue placeholder="Select priority" />
                                                            </SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="low">Low - General question</SelectItem>
                                                                <SelectItem value="medium">Medium - Affecting workflow</SelectItem>
                                                                <SelectItem value="high">High - Cannot use platform</SelectItem>
                                                                <SelectItem value="urgent">Urgent - Data at risk</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                        {errors.priority && (
                                                            <p className="text-sm text-destructive">{errors.priority.message}</p>
                                                        )}
                                                    </div>
                                                </div>

                                                <div className="space-y-2">
                                                    <Label htmlFor="subject">Subject</Label>
                                                    <Input id="subject" placeholder="Brief description of the issue" {...register("subject")} />
                                                    {errors.subject && (
                                                        <p className="text-sm text-destructive">{errors.subject.message}</p>
                                                    )}
                                                </div>

                                                <div className="space-y-2">
                                                    <Label htmlFor="description">Detailed Description</Label>
                                                    <Textarea
                                                        id="description"
                                                        placeholder="Please describe the issue in detail. Include steps to reproduce, expected behavior, and what actually happened..."
                                                        rows={5}
                                                        {...register("description")}
                                                    />
                                                    {errors.description && (
                                                        <p className="text-sm text-destructive">{errors.description.message}</p>
                                                    )}
                                                </div>

                                                <div className="space-y-2">
                                                    <Label htmlFor="environment">Environment Details</Label>
                                                    <Input
                                                        id="environment"
                                                        placeholder="Browser, OS, device (e.g., Chrome 120, Windows 11, Desktop)"
                                                        {...register("environment")}
                                                    />
                                                    {errors.environment && (
                                                        <p className="text-sm text-destructive">{errors.environment.message}</p>
                                                    )}
                                                </div>

                                                <Button type="submit" className="w-full" disabled={isSubmitting}>
                                                    {isSubmitting ? "Submitting..." : "Submit Support Ticket"}
                                                </Button>
                                            </form>
                                        )}
                                    </CardContent>
                                </Card>

                                {/* Response Time Info */}
                                <div className="mt-6 grid gap-4 sm:grid-cols-2">
                                    <Card className="border-border/50 bg-card/50">
                                        <CardContent className="flex items-center gap-3 p-4">
                                            <Clock className="h-5 w-5 text-primary" />
                                            <div>
                                                <p className="font-medium text-foreground">Response Time</p>
                                                <p className="text-sm text-muted-foreground">Within 24 hours</p>
                                            </div>
                                        </CardContent>
                                    </Card>
                                    <Card className="border-border/50 bg-card/50">
                                        <CardContent className="flex items-center gap-3 p-4">
                                            <AlertCircle className="h-5 w-5 text-orange-500" />
                                            <div>
                                                <p className="font-medium text-foreground">Critical Issues</p>
                                                <p className="text-sm text-muted-foreground">4-hour response</p>
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>
                            </div>

                            {/* FAQ Section */}
                            <div>
                                <h2 className="text-2xl font-bold text-foreground">Common Issues</h2>
                                <p className="mt-2 text-muted-foreground">
                                    Quick solutions to frequently reported problems
                                </p>

                                <Accordion type="single" collapsible className="mt-6">
                                    {faqItems.map((item) => (
                                        <AccordionItem key={item.question} value={item.question.toLowerCase().replace(/[^a-z0-9]+/g, "-")}>
                                            <AccordionTrigger className="text-left text-foreground">
                                                {item.question}
                                            </AccordionTrigger>
                                            <AccordionContent className="text-muted-foreground">
                                                {item.answer}
                                            </AccordionContent>
                                        </AccordionItem>
                                    ))}
                                </Accordion>

                                {/* Additional Resources */}
                                <Card className="mt-8 border-border/50 bg-card/50">
                                    <CardHeader>
                                        <CardTitle className="text-lg">Need More Help?</CardTitle>
                                        <CardDescription>
                                            Check out these additional resources
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        <a
                                            href="/documentation"
                                            className="flex items-center justify-between rounded-lg border border-border/50 p-3 transition-colors hover:bg-muted/50"
                                        >
                                            <span className="font-medium text-foreground">Documentation</span>
                                            <Badge variant="secondary">Guides</Badge>
                                        </a>
                                        <a
                                            href="/api-docs"
                                            className="flex items-center justify-between rounded-lg border border-border/50 p-3 transition-colors hover:bg-muted/50"
                                        >
                                            <span className="font-medium text-foreground">API Reference</span>
                                            <Badge variant="secondary">Developers</Badge>
                                        </a>
                                        <a
                                            href="/live-chat"
                                            className="flex items-center justify-between rounded-lg border border-border/50 p-3 transition-colors hover:bg-muted/50"
                                        >
                                            <span className="font-medium text-foreground">Live Chat</span>
                                            <Badge variant="outline" className="border-green-500/50 text-green-600">Online</Badge>
                                        </a>
                                        <div className="rounded-lg border border-border/50 p-3">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <PhoneCall className="h-4 w-4 text-primary" />
                                                    <span className="font-medium text-foreground">Voice AI Support</span>
                                                </div>
                                                {voiceAgentStatus === "active" ? (
                                                    <Badge variant="outline" className="border-green-500/50 text-green-600">
                                                        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-green-500" />
                                                        Active
                                                    </Badge>
                                                ) : voiceAgentStatus === "checking" ? (
                                                    <Badge variant="outline" className="text-muted-foreground">Checking...</Badge>
                                                ) : (
                                                    <Badge variant="outline" className="text-muted-foreground">
                                                        {voiceAgentStatus === "ready" ? "Click to start" : "Unavailable"}
                                                    </Badge>
                                                )}
                                            </div>
                                            <p className="mt-1 text-xs text-muted-foreground">
                                                Llama 3.3 70B-powered voice assistant for tire & technical support
                                            </p>
                                            {voiceAgentStatus === "active" ? (
                                                <p className="mt-2 text-xs font-medium text-primary">
                                                    {voiceAgentName} — Call via your OmniDimension dashboard
                                                </p>
                                            ) : voiceAgentStatus === "ready" || voiceAgentStatus === "unavailable" ? (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="mt-2 w-full gap-2"
                                                    onClick={handleInitVoiceAgent}
                                                    disabled={isInitializing}
                                                >
                                                    <Mic className="h-3.5 w-3.5" />
                                                    {isInitializing ? "Starting Voice Agent..." : "Initialize Voice AI Agent"}
                                                </Button>
                                            ) : null}
                                            {voiceAgentStatus === "error" && (
                                                <p className="mt-2 text-xs text-destructive">
                                                    Failed to initialize. Check OMNIDIM_API_KEY in .env
                                                </p>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
            <Footer />
        </div>
    )
}
