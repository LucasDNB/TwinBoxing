/*
 * Prueba de humo de W1 (01 §7, riesgo 3): monta cada componente instalado de Animate UI.
 *
 * Existe porque Animate UI copia el codigo al repo pero depende de `motion`, que publico la
 * 14 el 02-10. Si alguien actualiza motion y un componente deja de montar, esto falla antes
 * que una pantalla.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import motionPkg from "motion/package.json";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/animate-ui/components/radix/accordion";
import { AlertDialog, AlertDialogContent, AlertDialogTitle, AlertDialogDescription } from "@/components/animate-ui/components/radix/alert-dialog";
import { Checkbox } from "@/components/animate-ui/components/radix/checkbox";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/animate-ui/components/radix/dropdown-menu";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/animate-ui/components/radix/popover";
import { Progress } from "@/components/animate-ui/components/radix/progress";
import { Sheet, SheetContent, SheetTitle, SheetDescription } from "@/components/animate-ui/components/radix/sheet";
import { SidebarProvider, Sidebar, SidebarContent } from "@/components/animate-ui/components/radix/sidebar";
import { Switch } from "@/components/animate-ui/components/radix/switch";
import { Tabs, TabsContent, TabsContents, TabsList, TabsTrigger } from "@/components/animate-ui/components/radix/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/animate-ui/components/radix/tooltip";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { CountingNumber } from "@/components/animate-ui/primitives/texts/counting-number";
import { Fade } from "@/components/animate-ui/primitives/effects/fade";
import { AutoHeight } from "@/components/animate-ui/primitives/effects/auto-height";
import { Upload } from "@/components/animate-ui/icons/upload";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

describe("Animate UI: cada componente monta", () => {
  it("motion sigue fijado en la 12.43.0", () => {
    expect(motionPkg.version).toBe("12.43.0");
  });

  it("accordion, tabs, toggle-group", () => {
    render(
      <>
        <Accordion type="single" collapsible>
          <AccordionItem value="a">
            <AccordionTrigger>abrir</AccordionTrigger>
            <AccordionContent>adentro</AccordionContent>
          </AccordionItem>
        </Accordion>
        <Tabs defaultValue="uno">
          <TabsList>
            <TabsTrigger value="uno">Uno</TabsTrigger>
            <TabsTrigger value="dos">Dos</TabsTrigger>
          </TabsList>
          <TabsContents>
            <TabsContent value="uno">contenido uno</TabsContent>
            <TabsContent value="dos">contenido dos</TabsContent>
          </TabsContents>
        </Tabs>
        <ToggleGroup type="single" defaultValue="A">
          <ToggleGroupItem value="A">A</ToggleGroupItem>
          <ToggleGroupItem value="B">B</ToggleGroupItem>
        </ToggleGroup>
      </>,
    );
    expect(screen.getByText("abrir")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Uno" })).toBeInTheDocument();
    expect(screen.getByText("contenido uno")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "A" })).toBeInTheDocument();
  });

  it("dialogos y hojas abiertos", () => {
    render(
      <>
        <AlertDialog open>
          <AlertDialogContent>
            <AlertDialogTitle>confirmar</AlertDialogTitle>
            <AlertDialogDescription>seguro</AlertDialogDescription>
          </AlertDialogContent>
        </AlertDialog>
        <Sheet open>
          <SheetContent side="bottom">
            <SheetTitle>hoja</SheetTitle>
            <SheetDescription>desc</SheetDescription>
          </SheetContent>
        </Sheet>
      </>,
    );
    expect(screen.getByText("confirmar")).toBeInTheDocument();
    expect(screen.getByText("hoja")).toBeInTheDocument();
  });

  it("popover, dropdown, tooltip abiertos", () => {
    render(
      <>
        <Popover open>
          <PopoverTrigger>p</PopoverTrigger>
          <PopoverContent>pop</PopoverContent>
        </Popover>
        <DropdownMenu open>
          <DropdownMenuTrigger>d</DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem>item</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Tooltip open>
          <TooltipTrigger>t</TooltipTrigger>
          <TooltipContent>tip</TooltipContent>
        </Tooltip>
      </>,
    );
    expect(screen.getByText("pop")).toBeInTheDocument();
    expect(screen.getByText("item")).toBeInTheDocument();
    expect(screen.getAllByText("tip").length).toBeGreaterThan(0);
  });

  it("controles sueltos", () => {
    render(
      <>
        <Progress value={40} />
        <Checkbox aria-label="casilla" />
        <Switch aria-label="interruptor" />
        <Button>boton</Button>
        <CountingNumber number={42} />
        <Fade>fade</Fade>
        <AutoHeight>alto</AutoHeight>
        <Upload />
        <Label htmlFor="i">campo</Label>
        <Input id="i" />
        <Card>tarjeta</Card>
        <Badge>insignia</Badge>
        <Skeleton className="h-4" />
        <SidebarProvider>
          <Sidebar collapsible="none">
            <SidebarContent>lateral</SidebarContent>
          </Sidebar>
        </SidebarProvider>
      </>,
    );
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "casilla" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "interruptor" })).toBeInTheDocument();
    expect(screen.getByText("lateral")).toBeInTheDocument();
  });
});
