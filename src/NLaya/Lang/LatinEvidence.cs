namespace NLaya.Lang;

/// <summary>What the Latin-script language guess saw (Python <c>latin_profile</c>).</summary>
internal sealed record LatinEvidence(string? Language, int EnglishHits, double DiacriticRate, bool LooksNonEnglish);
