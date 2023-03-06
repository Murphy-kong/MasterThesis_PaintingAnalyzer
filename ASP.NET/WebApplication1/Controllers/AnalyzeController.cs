using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;
using IronPython.Hosting;
using Microsoft.Scripting;
using Microsoft.Scripting.Hosting;

namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AnalyzeController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;

        public AnalyzeController(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }

        // GET: api/Analyze
        [HttpGet]
        public async Task<ActionResult<IEnumerable<AnalyzeModel>>> GetAnalyzes()
        {
            if (_context.Analyzes == null)
            {
                return NotFound();
            }
            return await _context.Analyzes
                .Include(c => c.Session_Artist_Accuracy)
                .Include(c => c.Session_Genre_Accuracy)
                .Include(c => c.Session_Style_Accuracy)
                .ToListAsync();
        }

        // GET: api/Analyze/5
        [HttpGet("{id}"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<AnalyzeModel>> GetAnalyzeModel(int id)
        {
            if (_context.Analyzes == null)
            {
                return NotFound();
            }
            var analyzeModel = await _context.Analyzes
                .Include(c => c.Session_Artist_Accuracy)
                .Include(c => c.Session_Genre_Accuracy)
                .Include(c => c.Session_Style_Accuracy)
                .Where(c => c.AnalyzeID == id)
                .FirstOrDefaultAsync(); 

            if (analyzeModel == null)
            {
                return NotFound();
            }

            return analyzeModel;
        }



        // PUT: api/Analyze/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutAnalyzeModel(int id, AnalyzeModel analyzeModel)
        {
            if (id != analyzeModel.AnalyzeID)
            {
                return BadRequest();
            }

            _context.Entry(analyzeModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!AnalyzeModelExists(id))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }

        // POST: api/Analyze
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        public async Task<ActionResult<AnalyzeModel>> PostAnalyzeModel(AnalyzeModel analyzeModel)
        {
            if (_context.Analyzes == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Analyzes'  is null.");
            }
       
           if (analyzeModel.ImageID == null)
            {
                var image = await _context.Images
               .Where(c => c.ImageID == analyzeModel.ImageID)
               .FirstOrDefaultAsync();
            }
            else
                return NotFound();



            _context.Analyzes.Add(analyzeModel);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetAnalyzeModel", new { id = analyzeModel.AnalyzeID }, analyzeModel);
        }

        // POST: api/Analyze
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost("AddArtist_to_Analyze")]
        public async Task<ActionResult<AnalyzeModel>> AddArtist_to_Analyze(int id_artist, int id_analyze)
        {
            if (_context.Analyzes == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Analyzes'  is null.");
            }

                var analyze = await _context.Analyzes
               .Where(c => c.AnalyzeID == id_analyze)
               .FirstOrDefaultAsync();

            if(analyze == null)
                return NotFound();

            var artist = await _context.Session_Artist_Accuracy
                                        .FindAsync(id_analyze);
            if (artist == null)
                return NotFound();

            analyze.Session_Artist_Accuracy.Add(artist);
            await _context.SaveChangesAsync();
            return analyze;
        }

        [HttpPost("AddGenre_to_Analyze")]
        public async Task<ActionResult<AnalyzeModel>> AddGenre_to_Analyze(int id_genre, int id_analyze)
        {
            if (_context.Analyzes == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Analyzes'  is null.");
            }

            var analyze = await _context.Analyzes
           .Where(c => c.AnalyzeID == id_analyze)
           .FirstOrDefaultAsync();

            if (analyze == null)
                return NotFound();

            var style = await _context.Session_Genre_Accuracy
                                        .FindAsync(id_analyze);
            if (style == null)
                return NotFound();

            analyze.Session_Genre_Accuracy.Add(style);
            await _context.SaveChangesAsync();
            return analyze;
        }

        [HttpPost("AddStyle_to_Analyze")]
        public async Task<ActionResult<AnalyzeModel>> AddStyle_to_Analyze(int id_style, int id_analyze)
        {
            if (_context.Analyzes == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Analyzes'  is null.");
            }

            var analyze = await _context.Analyzes
           .Where(c => c.AnalyzeID == id_analyze)
           .FirstOrDefaultAsync();

            if (analyze == null)
                return NotFound();

            var style = await _context.Session_Style_Accuracy
                                        .FindAsync(id_analyze);
            if (style == null)
                return NotFound();

            analyze.Session_Style_Accuracy.Add(style);
            await _context.SaveChangesAsync();
            return analyze;
        }

        [HttpPost("Get_data_From_Model"), Authorize(Roles = "Admin,Registered User")]
        public async Task<IActionResult> Get_data_From_Model(AnalyzeDto analyze)
        {
            var image = await _context.Images
               .Where(c => c.ImageID == analyze.ImageID)
               .FirstOrDefaultAsync();
            if (image == null)
                return NotFound();


            AnalyzeModel newanalyze = new AnalyzeModel();
            newanalyze.ImageID = analyze.ImageID;
            newanalyze.ReleaseDate = analyze.ReleaseDate;
            newanalyze.ML_Model = analyze.ML_Model;
            _context.Analyzes.Add(newanalyze);
            await _context.SaveChangesAsync();

            Session_Artist_AccuracyModel artist = new Session_Artist_AccuracyModel();
            artist.ArtistID = 0;
            artist.ArtistName = analyze.result_artist;
            artist.Accuracy = analyze.acc_artist;
            artist.AnalyzeID = newanalyze.AnalyzeID;

            _context.Session_Artist_Accuracy.Add(artist);

            Session_Style_AccuracyModel style = new Session_Style_AccuracyModel();
            style.StyleID = 0;
            style.StyleName = analyze.result_style;
            style.Accuracy = analyze.acc_style;
            style.AnalyzeID = newanalyze.AnalyzeID;

            _context.Session_Style_Accuracy.Add(style);

            Session_Genre_AccuracyModel genre = new Session_Genre_AccuracyModel();
            genre.GenreID = 0;
            genre.GenreName = analyze.result_genre;
            genre.Accuracy = analyze.acc_genre;
            genre.AnalyzeID = newanalyze.AnalyzeID;

            _context.Session_Genre_Accuracy.Add(genre);
            await _context.SaveChangesAsync();

            Console.WriteLine(newanalyze.AnalyzeID);


            return Ok(newanalyze);
        }

        // DELETE: api/Analyze/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteAnalyzeModel(int id)
        {
            if (_context.Analyzes == null)
            {
                return NotFound();
            }
            var analyzeModel = await _context.Analyzes.FindAsync(id);
            if (analyzeModel == null)
            {
                return NotFound();
            }

            _context.Analyzes.Remove(analyzeModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool AnalyzeModelExists(int id)
        {
            return (_context.Analyzes?.Any(e => e.AnalyzeID == id)).GetValueOrDefault();
        }
    }
}
