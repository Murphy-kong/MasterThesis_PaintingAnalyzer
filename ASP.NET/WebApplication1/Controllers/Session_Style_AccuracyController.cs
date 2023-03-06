using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;
using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;

namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class Session_Style_AccuracyController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;

        public Session_Style_AccuracyController(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }

        // GET: api/Session_Style_Accuracy
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Session_Style_AccuracyModel>>> GetSession_Style_Accuracy()
        {
          if (_context.Session_Style_Accuracy == null)
          {
              return NotFound();
          }
            return await _context.Session_Style_Accuracy.ToListAsync();
        }

        // GET: api/Session_Style_Accuracy/5
        [HttpGet("{id}"), Authorize(Roles = "Admin")]
        public async Task<ActionResult<Session_Style_AccuracyModel>> GetSession_Style_AccuracyModel(int id)
        {
          if (_context.Session_Style_Accuracy == null)
          {
              return NotFound();
          }
            var session_Style_AccuracyModel = await _context.Session_Style_Accuracy.FindAsync(id);

            if (session_Style_AccuracyModel == null)
            {
                return NotFound();
            }
      
            return session_Style_AccuracyModel;
        }

        // PUT: api/Session_Style_Accuracy/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutSession_Style_AccuracyModel(int id, Session_Style_AccuracyModel session_Style_AccuracyModel)
        {
            if (id != session_Style_AccuracyModel.StyleID)
            {
                return BadRequest();
            }

            _context.Entry(session_Style_AccuracyModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!Session_Style_AccuracyModelExists(id))
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

        // POST: api/Session_Style_Accuracy
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        public async Task<ActionResult<Session_Style_AccuracyModel>> PostSession_Style_AccuracyModel(Session_Style_AccuracyModel session_Style_AccuracyModel)
        {
          if (_context.Session_Style_Accuracy == null)
          {
              return Problem("Entity set 'PaintingAnalyzerDbContext.Session_Style_Accuracy'  is null.");
          }
            _context.Session_Style_Accuracy.Add(session_Style_AccuracyModel);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetSession_Style_AccuracyModel", new { id = session_Style_AccuracyModel.StyleID }, session_Style_AccuracyModel);
        }

        // DELETE: api/Session_Style_Accuracy/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteSession_Style_AccuracyModel(int id)
        {
            if (_context.Session_Style_Accuracy == null)
            {
                return NotFound();
            }
            var session_Style_AccuracyModel = await _context.Session_Style_Accuracy.FindAsync(id);
            if (session_Style_AccuracyModel == null)
            {
                return NotFound();
            }

            _context.Session_Style_Accuracy.Remove(session_Style_AccuracyModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool Session_Style_AccuracyModelExists(int id)
        {
            return (_context.Session_Style_Accuracy?.Any(e => e.StyleID == id)).GetValueOrDefault();
        }
    }
}
