namespace BookRecommender.Core.Models;

public sealed class BookRecord
{
    public string Title { get; init; } = string.Empty;
    public string Authors { get; init; } = string.Empty;
    public string Categories { get; init; } = string.Empty;
    public string PublishedYear { get; init; } = string.Empty;

    public string Combined =>
        $"{Title} {Authors} {Categories} {PublishedYear}";
}
