using Microsoft.ML.Data;

namespace BookRecommender.Core.Models;

/// <summary>
/// Helper class used after transformation to pick out the vector.
/// </summary>
internal sealed class FeaturesRow
{
    [VectorType] public float[] Features { get; set; } = Array.Empty<float>();
}
