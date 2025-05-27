using BookRecommender.Core.Models;
using FuzzySharp;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;

namespace BookRecommender.Core.Services;

public sealed class RecommendationEngine
{
    private const string ModelPath = "tfidf.zip";
    private const string MatrixPath = "similarity.bin";

    private readonly MLContext _ml = new();

    private readonly BookRecord[] _books;
    private readonly float[][] _vectors;
    private readonly float[,] _similarity;
    private readonly ConcurrentDictionary<string, int> _titleIndex = new(StringComparer.OrdinalIgnoreCase);

    // ──────────────────────────────────────────────────────────────────────────
    public RecommendationEngine(string csvPath)
    {
        // 0. CSV → POCOs
        var dataView = _ml.Data.LoadFromTextFile<BookCsvRow>(
                                path: csvPath,
                                separatorChar: ',',
                                hasHeader: true,
                                allowQuoting: true,     // ← keeps commas inside quoted fields
                                trimWhitespace: true);



        _books = _ml.Data
                    .CreateEnumerable<BookCsvRow>(dataView, reuseRowObject: false)
                    .Select(r => new BookRecord
                    {
                        Title = r.Title,
                        Authors = r.Authors,
                        Categories = r.Categories,
                        PublishedYear = r.PublishedYear
                    })
                    .ToArray();

        for (int i = 0; i < _books.Length; i++)
            _titleIndex[_books[i].Title] = i;

        // 1. TF-IDF  - load or fit
        var pipeline = _ml.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(BookRecord.Combined));

        ITransformer model;
        IDataView featureDv;

        bool warm = File.Exists(ModelPath) && File.Exists(MatrixPath);

        if (warm)
        {
            model = _ml.Model.Load(ModelPath, out _);
            featureDv = model.Transform(_ml.Data.LoadFromEnumerable(_books));
            _similarity = LoadMatrix(MatrixPath);
        }
        else
        {
            model = pipeline.Fit(_ml.Data.LoadFromEnumerable(_books));
            featureDv = model.Transform(_ml.Data.LoadFromEnumerable(_books));
        }

        // 2. Extract vectors (needed for Recommend)
        _vectors = _ml.Data
                      .CreateEnumerable<FeaturesRow>(featureDv, reuseRowObject: false)
                      .Select(r => r.Features)
                      .ToArray();

        // 3. Build & cache similarity if first run
        if (!warm)
        {
            _similarity = BuildSimilarityMatrix(_vectors);
            _ml.Model.Save(model, featureDv.Schema, ModelPath);
            SaveMatrix(_similarity, MatrixPath);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    public IReadOnlyList<(string Title, float Score)> Recommend(string title, int top = 10)
    {
        if (!_titleIndex.TryGetValue(title, out var idx))
        {
            var best = Process.ExtractOne(title, _titleIndex.Keys);
            if (best is null) return Array.Empty<(string, float)>();
            idx = _titleIndex[best.Value];
        }

        return Enumerable.Range(0, _books.Length)
                         .Select(i => (Title: _books[i].Title, Score: _similarity[idx, i]))
                         .OrderByDescending(t => t.Score)
                         .Skip(1)        // drop self
                         .Take(top)
                         .ToList();
    }

    // ──────────────────────────────────────────────────────────────────────────
    private static float[,] BuildSimilarityMatrix(float[][] v)
    {
        int n = v.Length;
        var sim = new float[n, n];

        // ❶ cache norms
        var norms = new float[n];
        for (int i = 0; i < n; i++)
            norms[i] = MathF.Sqrt(v[i].Sum(x => x * x)) + 1e-10f;

        // ❷ fill upper-triangle in parallel
        Parallel.For(0, n, i =>
        {
            var vi = v[i];
            var ni = norms[i];

            for (int j = i; j < n; j++)
            {
                var score = DotSIMD(vi, v[j]) / (ni * norms[j]);
                sim[i, j] = sim[j, i] = score;   // symmetry
            }
        });

        return sim;
    }

    // 64-bit-wide SIMD dot (works on both AVX & SSE2 CPUs)
    private static float DotSIMD(float[] a, float[] b)
    {
        var spanA = MemoryMarshal.Cast<float, Vector<float>>(a);
        var spanB = MemoryMarshal.Cast<float, Vector<float>>(b);

        Vector<float> acc = Vector<float>.Zero;
        for (int i = 0; i < spanA.Length; i++)
            acc += spanA[i] * spanB[i];

        float dot = Vector.Dot(acc, Vector<float>.One);

        // tail
        int rem = a.Length - spanA.Length * Vector<float>.Count;
        for (int k = a.Length - rem; k < a.Length; k++)
            dot += a[k] * b[k];

        return dot;
    }

    // ───────────────────────── persistence helpers ───────────────────────────
    private static void SaveMatrix(float[,] m, string path)
    {
        using var bw = new BinaryWriter(File.Create(path));
        int n = m.GetLength(0);
        bw.Write(n);
        for (int i = 0; i < n; i++)
            bw.Write(MemoryMarshal.Cast<float, byte>(m.GetRowSpan(i)));
    }

    private static float[,] LoadMatrix(string path)
    {
        using var br = new BinaryReader(File.OpenRead(path));
        int n = br.ReadInt32();
        var m = new float[n, n];
        for (int i = 0; i < n; i++)
            br.Read(MemoryMarshal.Cast<float, byte>(m.GetRowSpan(i)));
        return m;
    }
}

// quick Span helper
file static class Array2DExt
{
    public static Span<float> GetRowSpan(this float[,] m, int row) =>
        MemoryMarshal.CreateSpan(ref m[row, 0], m.GetLength(1));
}
