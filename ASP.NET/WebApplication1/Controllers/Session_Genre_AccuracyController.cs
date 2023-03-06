using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;

namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class Session_Genre_AccuracyController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;

        public Session_Genre_AccuracyController(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }

        // GET: api/Session_Genre_Accuracy
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Session_Genre_AccuracyModel>>> GetSession_Genre_Accuracy()
        {
          if (_context.Session_Genre_Accuracy == null)
          {
              return NotFound();
          }
            return await _context.Session_Genre_Accuracy.ToListAsync();
        }

        // GET: api/Session_Genre_Accuracy/5
        [HttpGet("{id}")]
        public async Task<ActionResult<Session_Genre_AccuracyModel>> GetSession_Genre_AccuracyModel(int id)
        {
          if (_context.Session_Genre_Accuracy == null)
          {
              return NotFound();
          }
            var session_Genre_AccuracyModel = await _context.Session_Genre_Accuracy.FindAsync(id);

            if (session_Genre_AccuracyModel == null)
            {
                return NotFound();
            }

            return session_Genre_AccuracyModel;
        }

        // PUT: api/Session_Genre_Accuracy/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutSession_Genre_AccuracyModel(int id, Session_Genre_AccuracyModel session_Genre_AccuracyModel)
        {
            if (id != session_Genre_AccuracyModel.GenreID)
            {
                return BadRequest();
            }

            _context.Entry(session_Genre_AccuracyModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!Session_Genre_AccuracyModelExists(id))
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

        // POST: api/Session_Genre_Accuracy
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        public async Task<ActionResult<Session_Genre_AccuracyModel>> PostSession_Genre_AccuracyModel(Session_Genre_AccuracyModel session_Genre_AccuracyModel)
        {
          if (_context.Session_Genre_Accuracy == null)
          {
              return Problem("Entity set 'PaintingAnalyzerDbContext.Session_Genre_Accuracy'  is null.");
          }
            _context.Session_Genre_Accuracy.Add(session_Genre_AccuracyModel);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetSession_Genre_AccuracyModel", new { id = session_Genre_AccuracyModel.GenreID }, session_Genre_AccuracyModel);
        }

        // DELETE: api/Session_Genre_Accuracy/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteSession_Genre_AccuracyModel(int id)
        {
            if (_context.Session_Genre_Accuracy == null)
            {
                return NotFound();
            }
            var session_Genre_AccuracyModel = await _context.Session_Genre_Accuracy.FindAsync(id);
            if (session_Genre_AccuracyModel == null)
            {
                return NotFound();
            }

            _context.Session_Genre_Accuracy.Remove(session_Genre_AccuracyModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool Session_Genre_AccuracyModelExists(int id)
        {
            return (_context.Session_Genre_Accuracy?.Any(e => e.GenreID == id)).GetValueOrDefault();
        }
    }
}
