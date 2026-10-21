// Build a nexus geometry from an init macro and run a visualisation macro on it
// (no interactive session): ./geom_export <init_macro> <vis_macro>
#include "NexusApp.h"
#include <G4UImanager.hh>
#include <G4UIExecutive.hh>
#include <G4VisExecutive.hh>
#include <G4String.hh>
#include <execinfo.h>
#include <csignal>
#include <unistd.h>

static void on_segv(int sig)
{
  void* frames[64];
  int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, STDERR_FILENO);
  _exit(128 + sig);
}

int main(int argc, char** argv)
{
  if (argc < 3) return 1;
  std::signal(SIGSEGV, on_segv);
  auto* app = new nexus::NexusApp(argv[1]);
  app->Initialize();
  char* uiargv[] = {argv[0]};
  auto* ui = new G4UIExecutive(1, uiargv, "tcsh");   // created, never started
  auto* vis = new G4VisExecutive("errors");
  vis->Initialize();
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[2]);
  delete vis;
  delete ui;
  return 0;
}
