using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class AnalyzeDto
    {
        public int? AnalyzeID { get; set; }

        public int? ImageID { get; set; }

        public string? ML_Model { get; set; }

        public float acc_artist { get; set; }

        public string result_artist { get; set; }

        public float acc_style { get; set; }

        public string result_style { get; set; }

        public float acc_genre { get; set; }

        public string result_genre { get; set; }


        [DataType(DataType.Date)]
        public DateTime ReleaseDate { get; set; }

    }
}
