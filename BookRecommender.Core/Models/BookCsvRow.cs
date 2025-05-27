using Microsoft.ML.Data;

namespace BookRecommender.Core.Models;

public sealed class BookCsvRow
{
    [LoadColumn(2)] public string Title { get; set; } = string.Empty;  // ← 2
    [LoadColumn(4)] public string Authors { get; set; } = string.Empty;  // ← 4
    [LoadColumn(5)] public string Categories { get; set; } = string.Empty;  // ← 5
    [LoadColumn(8)] public string PublishedYear { get; set; } = string.Empty;  // ← 8
}
