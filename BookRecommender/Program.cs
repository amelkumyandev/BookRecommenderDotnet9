using BookRecommender.Core.Services;
using Spectre.Console;
using System.Diagnostics;

AnsiConsole.MarkupLine("[cyan]Loading model, please wait…[/]");

var sw = Stopwatch.StartNew();
var engine = new RecommendationEngine("data.csv");
AnsiConsole.MarkupLine($"[grey](ready in {sw.Elapsed.TotalSeconds:N1} s)[/]\n");

AnsiConsole.MarkupLine("[bold yellow]Type a book title (or ENTER to quit).[/]");
while (true)
{
    AnsiConsole.Markup("[green]▶ [/]");
    var title = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(title)) break;

    var recs = engine.Recommend(title, 10);
    if (recs.Count == 0)
    {
        AnsiConsole.MarkupLine("[red]No match found.[/]");
        continue;
    }

    for (int i = 0; i < recs.Count; i++)
        AnsiConsole.MarkupLine($"{i + 1,2}. {recs[i].Title}  [grey](score {recs[i].Score:F3})[/]");
    AnsiConsole.WriteLine();
}
