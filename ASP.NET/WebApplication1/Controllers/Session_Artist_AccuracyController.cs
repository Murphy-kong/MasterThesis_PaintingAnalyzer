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
    public class Session_Artist_AccuracyController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;

        public Session_Artist_AccuracyController(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }

        // GET: api/Session_Artist_Accuracy
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Session_Artist_AccuracyModel>>> GetSession_Artist_Accuracy()
        {
          if (_context.Session_Artist_Accuracy == null)
          {
              return NotFound();
          }
            return await _context.Session_Artist_Accuracy.ToListAsync();
        }

        // GET: api/Session_Artist_Accuracy/5
        [HttpGet("{id}")]
        public async Task<ActionResult<Session_Artist_AccuracyModel>> GetSession_Artist_AccuracyModel(int id)
        {
          if (_context.Session_Artist_Accuracy == null)
          {
              return NotFound();
          }
            var session_Artist_AccuracyModel = await _context.Session_Artist_Accuracy.FindAsync(id);

            if (session_Artist_AccuracyModel == null)
            {
                return NotFound();
            }

            return session_Artist_AccuracyModel;
        }

        // PUT: api/Session_Artist_Accuracy/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutSession_Artist_AccuracyModel(int id, Session_Artist_AccuracyModel session_Artist_AccuracyModel)
        {
            if (id != session_Artist_AccuracyModel.ArtistID)
            {
                return BadRequest();
            }

            _context.Entry(session_Artist_AccuracyModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!Session_Artist_AccuracyModelExists(id))
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

        // POST: api/Session_Artist_Accuracy
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        public async Task<ActionResult<Session_Artist_AccuracyModel>> PostSession_Artist_AccuracyModel(Session_Artist_AccuracyModel session_Artist_AccuracyModel)
        {
          if (_context.Session_Artist_Accuracy == null)
          {
              return Problem("Entity set 'PaintingAnalyzerDbContext.Session_Artist_Accuracy'  is null.");
          }
            _context.Session_Artist_Accuracy.Add(session_Artist_AccuracyModel);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetSession_Artist_AccuracyModel", new { id = session_Artist_AccuracyModel.ArtistID }, session_Artist_AccuracyModel);
        }

        // DELETE: api/Session_Artist_Accuracy/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteSession_Artist_AccuracyModel(int id)
        {
            if (_context.Session_Artist_Accuracy == null)
            {
                return NotFound();
            }
            var session_Artist_AccuracyModel = await _context.Session_Artist_Accuracy.FindAsync(id);
            if (session_Artist_AccuracyModel == null)
            {
                return NotFound();
            }

            _context.Session_Artist_Accuracy.Remove(session_Artist_AccuracyModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool Session_Artist_AccuracyModelExists(int id)
        {
            return (_context.Session_Artist_Accuracy?.Any(e => e.ArtistID == id)).GetValueOrDefault();
        }
    }
}
