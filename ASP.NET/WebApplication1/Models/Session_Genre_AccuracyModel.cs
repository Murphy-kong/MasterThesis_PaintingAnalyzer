using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class Session_Genre_AccuracyModel
    {
        [Key]
        public int GenreID { get; set; }

        public string GenreName { get; set; }

        public float Accuracy { get; set; }

        public int? AnalyzeID { get; set; }

        [JsonIgnore]

        public AnalyzeModel? Analyze { get; set; }
    }
}
